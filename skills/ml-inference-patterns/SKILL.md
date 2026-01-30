---
name: ml-inference-patterns
description: ML inference patterns - batching, caching, torch.no_grad, model compilation/export, latency vs throughput optimization.
---

# ML Inference Patterns

Patterns for efficient model inference.

## Basic Inference

### Inference Wrapper

```python
import torch
from torch import Tensor


class Predictor:
    """Inference wrapper for models."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def predict(self, inputs: dict[str, Tensor]) -> Tensor:
        """Run inference on inputs.

        Args:
            inputs: Dictionary with input tensors.

        Returns:
            Model output tensor.
        """
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs)

    @torch.no_grad()
    def predict_proba(self, inputs: dict[str, Tensor]) -> Tensor:
        """Get probability predictions."""
        logits = self.predict(inputs)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_class(self, inputs: dict[str, Tensor]) -> Tensor:
        """Get class predictions."""
        logits = self.predict(inputs)
        return logits.argmax(dim=-1)
```

### Using torch.inference_mode

```python
# torch.inference_mode is faster than torch.no_grad
# Use for pure inference (no gradient computation needed)

@torch.inference_mode()
def predict_fast(model: torch.nn.Module, inputs: Tensor) -> Tensor:
    """Fast inference without gradient tracking."""
    return model(inputs)


# torch.no_grad still needed when you might need gradients later
# e.g., for gradient-based explainability
```

## Batched Inference

### Batch Processing

```python
from torch.utils.data import DataLoader


class BatchPredictor:
    """Predictor with batched inference support."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict_batch(self, inputs: list[dict]) -> list[Tensor]:
        """Predict on a batch of inputs."""
        # Create temporary dataset/dataloader
        loader = DataLoader(inputs, batch_size=self.batch_size)

        all_outputs = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            all_outputs.append(outputs.cpu())

        return torch.cat(all_outputs, dim=0)
```

### Dynamic Batching

```python
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Request:
    """Inference request with future for response."""

    inputs: dict[str, Tensor]
    future: asyncio.Future


class DynamicBatcher:
    """Accumulate requests and process in batches."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: deque[Request] = deque()
        self._lock = asyncio.Lock()

    async def predict(self, inputs: dict[str, Tensor]) -> Tensor:
        """Submit request and wait for result."""
        future = asyncio.get_event_loop().create_future()
        request = Request(inputs=inputs, future=future)

        async with self._lock:
            self.queue.append(request)

            # Process if batch is full or timeout
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()

        return await future

    @torch.inference_mode()
    async def _process_batch(self) -> None:
        """Process accumulated requests."""
        if not self.queue:
            return

        # Collect batch
        batch_requests = []
        while self.queue and len(batch_requests) < self.max_batch_size:
            batch_requests.append(self.queue.popleft())

        # Stack inputs
        batched_inputs = self._collate([r.inputs for r in batch_requests])
        batched_inputs = {k: v.to(self.device) for k, v in batched_inputs.items()}

        # Run inference
        outputs = self.model(**batched_inputs)

        # Distribute results
        for i, request in enumerate(batch_requests):
            request.future.set_result(outputs[i].cpu())
```

## Caching

### LRU Cache for Embeddings

```python
from functools import lru_cache
import hashlib


class CachedEmbedder:
    """Embedder with caching for repeated inputs."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        cache_size: int = 1000,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._cache_size = cache_size
        self._cache: dict[str, Tensor] = {}

    def _hash_input(self, text: str) -> str:
        """Create hash key for input."""
        return hashlib.md5(text.encode()).hexdigest()

    @torch.inference_mode()
    def embed(self, text: str) -> Tensor:
        """Get embedding with caching."""
        key = self._hash_input(text)

        if key in self._cache:
            return self._cache[key]

        # Compute embedding
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        embedding = embedding.cpu()

        # Cache (with simple LRU eviction)
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = embedding
        return embedding
```

### Redis Caching (Production)

```python
import redis
import pickle
from typing import Optional


class RedisCachedPredictor:
    """Predictor with Redis caching."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl_seconds

    def _get_cache_key(self, inputs: dict) -> str:
        """Generate cache key from inputs."""
        # Use hash of serialized inputs
        serialized = pickle.dumps(inputs)
        return f"pred:{hashlib.sha256(serialized).hexdigest()}"

    @torch.inference_mode()
    def predict(self, inputs: dict[str, Tensor]) -> Tensor:
        """Predict with Redis caching."""
        cache_key = self._get_cache_key(inputs)

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return pickle.loads(cached)

        # Compute prediction
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        output = self.model(**inputs).cpu()

        # Cache result
        self.redis.setex(cache_key, self.ttl, pickle.dumps(output))

        return output
```

## Model Optimization

### torch.compile (PyTorch 2.0+)

```python
import torch


def optimize_model_compile(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model with torch.compile."""
    # mode options:
    # - "default": Good balance of speed and compile time
    # - "reduce-overhead": Minimizes Python overhead
    # - "max-autotune": Maximum performance, longer compile time

    compiled_model = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,  # Compile entire model as one graph
    )

    return compiled_model


# Usage
model = optimize_model_compile(model)

# First inference is slow (compilation)
# Subsequent inferences are much faster
```

### ONNX Export

```python
import torch.onnx


def export_to_onnx(
    model: torch.nn.Module,
    sample_input: Tensor,
    output_path: str,
    opset_version: int = 17,
) -> None:
    """Export model to ONNX format."""
    model.eval()

    torch.onnx.export(
        model,
        sample_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
        },
    )


# Usage
sample = torch.randn(1, 512)
export_to_onnx(model, sample, "model.onnx")
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np


class ONNXPredictor:
    """Predictor using ONNX Runtime."""

    def __init__(self, onnx_path: str, device: str = "cuda") -> None:
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Run inference with ONNX Runtime."""
        outputs = self.session.run(
            None,
            {self.input_name: inputs},
        )
        return outputs[0]
```

### TorchScript Export

```python
def export_to_torchscript(
    model: torch.nn.Module,
    sample_input: Tensor,
    output_path: str,
    method: str = "trace",
) -> None:
    """Export model to TorchScript."""
    model.eval()

    if method == "trace":
        # Tracing: records operations on sample input
        scripted = torch.jit.trace(model, sample_input)
    else:
        # Scripting: analyzes Python code
        scripted = torch.jit.script(model)

    scripted.save(output_path)


# Load and use
loaded_model = torch.jit.load("model.pt")
output = loaded_model(input_tensor)
```

## Latency vs Throughput

### Latency Optimization

```python
# For low latency (single request):

# 1. Use smaller batch size (1)
# 2. Use torch.inference_mode
# 3. Use torch.compile with reduce-overhead
# 4. Keep model in memory (don't reload)
# 5. Pin memory for GPU transfers

class LowLatencyPredictor:
    """Optimized for minimal latency."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = torch.compile(model, mode="reduce-overhead")
        self.model = self.model.to(device).eval()
        self.device = device

        # Warmup to compile
        dummy = torch.randn(1, 512).to(device)
        with torch.inference_mode():
            _ = self.model(dummy)

    @torch.inference_mode()
    def predict(self, inputs: Tensor) -> Tensor:
        inputs = inputs.to(self.device, non_blocking=True)
        return self.model(inputs)
```

### Throughput Optimization

```python
# For high throughput (many requests):

# 1. Use larger batch sizes
# 2. Use dynamic batching
# 3. Use multiple workers/GPUs
# 4. Use async processing

class HighThroughputPredictor:
    """Optimized for maximum throughput."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 64,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict_batch(self, inputs: list[Tensor]) -> list[Tensor]:
        """Process large batch for throughput."""
        results = []

        for i in range(0, len(inputs), self.batch_size):
            batch = torch.stack(inputs[i : i + self.batch_size])
            batch = batch.to(self.device)
            outputs = self.model(batch)
            results.extend(outputs.cpu().unbind(0))

        return results
```

## Memory Optimization

### Half Precision Inference

```python
class FP16Predictor:
    """Predictor using FP16 for reduced memory."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.half().to(device).eval()
        self.device = device

    @torch.inference_mode()
    def predict(self, inputs: Tensor) -> Tensor:
        inputs = inputs.half().to(self.device)
        return self.model(inputs).float()  # Convert back for compatibility
```

### Gradient Checkpointing (Large Models)

```python
# For very large models that don't fit in memory
# Trade compute for memory

from torch.utils.checkpoint import checkpoint


class MemoryEfficientModel(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # Use checkpointing for memory-heavy layers
        x = checkpoint(self.large_layer1, x, use_reentrant=False)
        x = checkpoint(self.large_layer2, x, use_reentrant=False)
        return self.output_layer(x)
```

## Quantization

### Dynamic Quantization

```python
def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    """Apply dynamic quantization for faster CPU inference."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8,
    )
    return quantized


# Useful for CPU inference with minimal accuracy loss
```

### Static Quantization (Advanced)

```python
# Static quantization requires calibration data
# More complex but can be more efficient

from torch.quantization import prepare, convert


def quantize_static(
    model: torch.nn.Module,
    calibration_loader: DataLoader,
) -> torch.nn.Module:
    """Apply static quantization with calibration."""
    model.eval()

    # Specify quantization config
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # Prepare model for calibration
    prepared = prepare(model)

    # Run calibration
    with torch.no_grad():
        for batch in calibration_loader:
            prepared(batch)

    # Convert to quantized model
    quantized = convert(prepared)

    return quantized
```

## Checklist

Before deploying inference:
- [ ] Model in eval mode
- [ ] torch.no_grad() or torch.inference_mode() used
- [ ] Batching strategy chosen (latency vs throughput)
- [ ] Caching implemented for repeated inputs
- [ ] Model optimized (compile/ONNX/quantize as needed)
- [ ] Memory usage acceptable
- [ ] Warmup performed for compiled models
- [ ] Error handling for edge cases

**Remember**: Optimize for your use case - low latency needs small batches, high throughput needs large batches and parallelism.
