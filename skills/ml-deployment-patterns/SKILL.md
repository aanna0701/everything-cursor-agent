---
name: ml-deployment-patterns
description: ML deployment patterns - FastAPI service, health checks, versioning, canary deployment, monitoring basics.
---

# ML Deployment Patterns

Patterns for deploying ML models as services.

## FastAPI Service

### Basic Service Structure

```python
# src/project/inference/server.py
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from project.models import TextClassifier
from project.inference.predictor import Predictor


# Request/Response models
class PredictRequest(BaseModel):
    """Prediction request."""

    text: str = Field(..., min_length=1, max_length=10000)
    return_proba: bool = Field(default=False)


class PredictResponse(BaseModel):
    """Prediction response."""

    label: int
    confidence: float
    probabilities: list[float] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# Global state
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier.from_pretrained(Path("models/production"))
    predictor = Predictor(model, device)

    yield

    # Cleanup
    predictor = None
    torch.cuda.empty_cache()


app = FastAPI(
    title="ML Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Run prediction on input text."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = predictor.predict_text(request.text)

        response = PredictResponse(
            label=result["label"],
            confidence=result["confidence"],
        )

        if request.return_proba:
            response.probabilities = result["probabilities"]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Running the Service

```bash
# Development
uvicorn project.inference.server:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn project.inference.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## Health Checks

### Comprehensive Health Check

```python
from dataclasses import dataclass
from datetime import datetime
import torch


@dataclass
class HealthStatus:
    """Detailed health status."""

    status: str  # "healthy", "degraded", "unhealthy"
    checks: dict[str, bool]
    details: dict[str, str]
    timestamp: str


async def detailed_health_check() -> HealthStatus:
    """Comprehensive health check."""
    checks = {}
    details = {}

    # Check model loaded
    checks["model_loaded"] = predictor is not None
    details["model_loaded"] = "Model loaded" if predictor else "Model not loaded"

    # Check GPU (if expected)
    if torch.cuda.is_available():
        try:
            # Simple GPU test
            _ = torch.tensor([1.0]).cuda()
            checks["gpu_available"] = True
            details["gpu"] = f"GPU: {torch.cuda.get_device_name(0)}"
        except Exception as e:
            checks["gpu_available"] = False
            details["gpu"] = f"GPU error: {e}"
    else:
        checks["gpu_available"] = True  # Not required
        details["gpu"] = "Running on CPU"

    # Check memory
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_ratio = memory_allocated / memory_total

        checks["memory_ok"] = memory_ratio < 0.9
        details["memory"] = f"{memory_allocated:.1f}GB / {memory_total:.1f}GB"

    # Determine overall status
    if all(checks.values()):
        status = "healthy"
    elif checks.get("model_loaded", False):
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthStatus(
        status=status,
        checks=checks,
        details=details,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health/detailed")
async def health_detailed():
    """Detailed health check for monitoring."""
    return await detailed_health_check()
```

### Kubernetes Probes

```python
@app.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe - is the process alive?"""
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe - is it ready for traffic?"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/health/startup")
async def startup_probe():
    """Kubernetes startup probe - has startup completed?"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Still loading")
    return {"status": "started"}
```

## Model Versioning

### Version Management

```python
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class ModelVersion:
    """Model version metadata."""

    version: str
    created_at: str
    metrics: dict[str, float]
    config: dict


class VersionedModelLoader:
    """Load models by version."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir

    def list_versions(self) -> list[str]:
        """List available model versions."""
        versions = []
        for path in self.models_dir.iterdir():
            if path.is_dir() and (path / "model.pt").exists():
                versions.append(path.name)
        return sorted(versions, reverse=True)

    def get_latest(self) -> str:
        """Get latest version."""
        versions = self.list_versions()
        if not versions:
            raise ValueError("No model versions found")
        return versions[0]

    def load_version(self, version: str) -> tuple[torch.nn.Module, ModelVersion]:
        """Load specific model version."""
        version_dir = self.models_dir / version

        # Load metadata
        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        version_info = ModelVersion(**metadata)

        # Load model
        model = TextClassifier(**version_info.config)
        model.load_state_dict(
            torch.load(version_dir / "model.pt", weights_only=True)
        )

        return model, version_info


# API endpoint for version info
@app.get("/model/version")
async def get_model_version():
    """Get current model version info."""
    return {
        "version": current_version.version,
        "created_at": current_version.created_at,
        "metrics": current_version.metrics,
    }
```

### A/B Testing Endpoint

```python
import random


class ABTestingPredictor:
    """Predictor with A/B testing support."""

    def __init__(
        self,
        model_a: Predictor,
        model_b: Predictor,
        traffic_split: float = 0.5,
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split

    def predict(self, text: str) -> dict:
        """Route prediction to A or B model."""
        use_b = random.random() < self.traffic_split

        model = self.model_b if use_b else self.model_a
        result = model.predict_text(text)

        # Add metadata for tracking
        result["model_variant"] = "B" if use_b else "A"

        return result
```

## Monitoring

### Metrics with Prometheus

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response


# Metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_DISTRIBUTION = Counter(
    "prediction_labels_total",
    "Distribution of predicted labels",
    ["label"],
)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain",
    )


# Instrumented prediction endpoint
import time


@app.post("/predict")
async def predict_with_metrics(request: PredictRequest):
    start_time = time.time()

    try:
        result = predictor.predict_text(request.text)

        # Record metrics
        REQUEST_COUNT.labels(status="success").inc()
        PREDICTION_DISTRIBUTION.labels(label=str(result["label"])).inc()

        return result

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)
```

### Structured Logging

```python
import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms

        return json.dumps(log_data)


# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger().addHandler(handler)
```

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

# Copy application code
COPY src/ src/
COPY models/ models/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["uvicorn", "project.inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-inference
  template:
    metadata:
      labels:
        app: ml-inference
    spec:
      containers:
        - name: inference
          image: my-registry/ml-inference:v1.0.0
          ports:
            - containerPort: 8000
          resources:
            requests:
              memory: "2Gi"
              cpu: "1"
              nvidia.com/gpu: 1
            limits:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: 1
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 5
          startupProbe:
            httpGet:
              path: /health/startup
              port: 8000
            failureThreshold: 30
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ml-inference
spec:
  selector:
    app: ml-inference
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-inference
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Checklist

Before deploying:
- [ ] Health check endpoints implemented
- [ ] Model version tracked
- [ ] Metrics exposed (latency, throughput, errors)
- [ ] Structured logging configured
- [ ] Input validation on all endpoints
- [ ] Error handling returns appropriate status codes
- [ ] Resource limits configured
- [ ] Graceful shutdown implemented
- [ ] Security: no secrets in code, HTTPS in production
- [ ] Load testing performed

**Remember**: A deployed model needs monitoring, versioning, and graceful handling of failures. Plan for the model to be wrong sometimes.
