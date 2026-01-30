---
name: security-review
description: Use this skill when adding authentication, handling user input, working with secrets, creating API endpoints, or implementing ML model deployment. Provides comprehensive security checklist and patterns for Python ML/DL projects.
---

# Security Review Skill (Python ML/DL)

This skill ensures all Python ML/DL code follows security best practices and identifies potential vulnerabilities.

## When to Activate

- Implementing authentication or authorization
- Handling user input or file uploads
- Creating new API endpoints (FastAPI, Flask)
- Working with secrets or credentials
- Deploying ML models to production
- Storing or transmitting sensitive data
- Integrating third-party APIs
- Handling model checkpoints and serialization

## Security Checklist

### 1. Secrets Management

#### ❌ NEVER Do This
```python
# Hardcoded secret
api_key = "sk-proj-xxxxx"
db_password = "password123"
```

#### ✅ ALWAYS Do This
```python
import os
from pathlib import Path

api_key = os.getenv("OPENAI_API_KEY")
db_url = os.getenv("DATABASE_URL")

# Verify secrets exist
if not api_key:
    raise ValueError("OPENAI_API_KEY not configured")
```

#### Verification Steps
- [ ] No hardcoded API keys, tokens, or passwords
- [ ] All secrets in environment variables
- [ ] `.env` files in .gitignore
- [ ] No secrets in git history
- [ ] Production secrets in secure storage (AWS Secrets Manager, Azure Key Vault, etc.)

### 2. Input Validation

#### Always Validate User Input
```python
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional


class CreateUserRequest(BaseModel):
    """User creation request with validation."""
    
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    
    @validator("name")
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


# Validate before processing
def create_user(request: CreateUserRequest) -> dict:
    """Create user with validated input."""
    # Request is already validated by Pydantic
    return {"success": True, "user_id": "..."}
```

#### File Upload Validation
```python
from pathlib import Path
from typing import Literal


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


def validate_file_upload(file_path: Path, file_size: int) -> bool:
    """Validate uploaded file."""
    # Size check
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File too large (max {MAX_FILE_SIZE} bytes)")
    
    # Extension check
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}")
    
    return True
```

#### Verification Steps
- [ ] All user inputs validated with Pydantic or similar
- [ ] File uploads restricted (size, type, extension)
- [ ] No direct use of user input in queries
- [ ] Whitelist validation (not blacklist)
- [ ] Error messages don't leak sensitive info

### 3. SQL Injection Prevention

#### ❌ NEVER Concatenate SQL
```python
# DANGEROUS - SQL Injection vulnerability
query = f"SELECT * FROM users WHERE email = '{user_email}'"
cursor.execute(query)
```

#### ✅ ALWAYS Use Parameterized Queries
```python
import sqlite3
from typing import Optional


def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email using parameterized query."""
    conn = sqlite3.connect("db.sqlite")
    cursor = conn.cursor()
    
    # Safe - parameterized query
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    
    if row:
        return {"id": row[0], "email": row[1]}
    return None
```

#### With SQLAlchemy
```python
from sqlalchemy.orm import Session
from models import User


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user using ORM (safe from SQL injection)."""
    return db.query(User).filter(User.email == email).first()
```

#### Verification Steps
- [ ] All database queries use parameterized queries
- [ ] No string formatting/f-strings in SQL
- [ ] ORM (SQLAlchemy) used correctly
- [ ] Raw SQL queries properly parameterized

### 4. Model Serialization Security

#### ❌ NEVER Use Unsafe Pickle
```python
# DANGEROUS - Can execute arbitrary code
import pickle

# Loading untrusted pickle files can execute malicious code
model = pickle.load(open("model.pkl", "rb"))
```

#### ✅ ALWAYS Use Safe Serialization
```python
import torch
from pathlib import Path


# Safe - PyTorch state_dict (weights only)
def save_model_safe(model: torch.nn.Module, path: Path) -> None:
    """Save model weights safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path, weights_only=True)


def load_model_safe(model: torch.nn.Module, path: Path) -> None:
    """Load model weights safely."""
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)


# For untrusted inputs, use safetensors
from safetensors.torch import load_file, save_file


def save_model_safetensors(model: torch.nn.Module, path: Path) -> None:
    """Save model using safetensors (safe for untrusted sources)."""
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, path)


def load_model_safetensors(model: torch.nn.Module, path: Path) -> None:
    """Load model using safetensors."""
    state_dict = load_file(path)
    model.load_state_dict(state_dict)
```

#### Verification Steps
- [ ] No `pickle.load()` on untrusted files
- [ ] Use `torch.load(weights_only=True)` for PyTorch
- [ ] Use `safetensors` for untrusted model sources
- [ ] Validate checkpoint file paths
- [ ] Check file signatures before loading

### 5. Authentication & Authorization

#### JWT Token Handling (FastAPI Example)
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os


security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token."""
    token = credentials.credentials
    secret = os.getenv("JWT_SECRET")
    
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured"
        )
    
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def require_admin(user: dict = Depends(verify_token)) -> dict:
    """Require admin role."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user
```

#### Verification Steps
- [ ] Tokens validated on every request
- [ ] Authorization checks before sensitive operations
- [ ] Role-based access control implemented
- [ ] Session management secure
- [ ] Token expiration enforced

### 6. Rate Limiting

#### API Rate Limiting (FastAPI)
```python
from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.get("/api/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: dict):
    """Rate-limited prediction endpoint."""
    # Model inference
    return {"prediction": "..."}
```

#### Verification Steps
- [ ] Rate limiting on all API endpoints
- [ ] Stricter limits on expensive operations (model inference)
- [ ] IP-based rate limiting
- [ ] User-based rate limiting (authenticated)

### 7. Sensitive Data Exposure

#### Logging
```python
import logging

logger = logging.getLogger(__name__)


# ❌ WRONG: Logging sensitive data
logger.info(f"User login: {email}, password: {password}")
logger.info(f"API key: {api_key}")

# ✅ CORRECT: Redact sensitive data
logger.info(f"User login: {email}, user_id: {user_id}")
logger.info(f"API key: {api_key[:8]}...")  # Only show prefix
```

#### Error Messages
```python
# ❌ WRONG: Exposing internal details
try:
    result = risky_operation()
except Exception as e:
    return {"error": str(e), "traceback": traceback.format_exc()}

# ✅ CORRECT: Generic error messages
try:
    result = risky_operation()
except Exception as e:
    logger.error("Internal error", exc_info=True)
    return {"error": "An error occurred. Please try again."}
```

#### Verification Steps
- [ ] No passwords, tokens, or secrets in logs
- [ ] Error messages generic for users
- [ ] Detailed errors only in server logs
- [ ] No stack traces exposed to users
- [ ] Model weights/predictions logged carefully (privacy)

### 8. Dependency Security

#### Regular Updates
```bash
# Check for vulnerabilities (WSL/bash)
uv pip list --outdated
pip-audit  # If installed

# Update dependencies
uv pip install --upgrade <package>

# Check security advisories
# Visit: https://pypi.org/project/<package>/
```

#### Lock Files
```bash
# ALWAYS commit lock files
git add uv.lock  # or requirements.txt

# Use in CI/CD for reproducible builds
uv sync  # Instead of uv pip install
```

#### Verification Steps
- [ ] Dependencies up to date
- [ ] No known vulnerabilities (check PyPI security advisories)
- [ ] Lock files committed (uv.lock, requirements.txt)
- [ ] Dependabot enabled on GitHub
- [ ] Regular security updates

### 9. ML-Specific Security

#### Model Checkpoint Validation
```python
from pathlib import Path
import hashlib


def validate_checkpoint(path: Path, expected_hash: str) -> bool:
    """Validate checkpoint file integrity."""
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    if file_hash != expected_hash:
        raise ValueError("Checkpoint hash mismatch - file may be corrupted or tampered")
    
    return True
```

#### Input Sanitization for Inference
```python
import numpy as np
import torch


def sanitize_input(input_data: np.ndarray, max_size: int = 10000) -> torch.Tensor:
    """Sanitize input for model inference."""
    # Check size
    if input_data.size > max_size:
        raise ValueError(f"Input too large (max {max_size} elements)")
    
    # Check for NaN/Inf
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        raise ValueError("Input contains NaN or Inf values")
    
    # Convert to tensor
    tensor = torch.from_numpy(input_data).float()
    
    # Check shape
    if len(tensor.shape) == 0:
        raise ValueError("Input must be at least 1D")
    
    return tensor
```

#### Verification Steps
- [ ] Model checkpoints validated before loading
- [ ] Input data sanitized (size, type, NaN/Inf checks)
- [ ] Output predictions validated (range, type)
- [ ] Model versioning and tracking
- [ ] A/B testing and canary deployments for model updates

## Security Testing

### Automated Security Tests
```python
import pytest
from fastapi.testclient import TestClient


def test_requires_authentication(client: TestClient):
    """Test that protected endpoints require authentication."""
    response = client.get("/api/protected")
    assert response.status_code == 401


def test_requires_admin_role(client: TestClient, user_token: str):
    """Test that admin endpoints require admin role."""
    response = client.get(
        "/api/admin",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 403


def test_rejects_invalid_input(client: TestClient):
    """Test input validation."""
    response = client.post(
        "/api/users",
        json={"email": "not-an-email"}
    )
    assert response.status_code == 422  # Pydantic validation error


def test_rate_limiting(client: TestClient):
    """Test rate limiting."""
    responses = [client.get("/api/predict") for _ in range(11)]
    rate_limited = [r for r in responses if r.status_code == 429]
    assert len(rate_limited) > 0
```

## Pre-Deployment Security Checklist

Before ANY production deployment:

- [ ] **Secrets**: No hardcoded secrets, all in env vars
- [ ] **Input Validation**: All user inputs validated (Pydantic)
- [ ] **SQL Injection**: All queries parameterized
- [ ] **Model Serialization**: Safe loading (weights_only=True, safetensors)
- [ ] **Authentication**: Proper token handling
- [ ] **Authorization**: Role checks in place
- [ ] **Rate Limiting**: Enabled on all endpoints
- [ ] **HTTPS**: Enforced in production
- [ ] **Error Handling**: No sensitive data in errors
- [ ] **Logging**: No sensitive data logged
- [ ] **Dependencies**: Up to date, no vulnerabilities
- [ ] **CORS**: Properly configured
- [ ] **File Uploads**: Validated (size, type)
- [ ] **Model Checkpoints**: Validated before loading
- [ ] **Input Sanitization**: All inference inputs sanitized

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security-warnings.html)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/security.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

---

**Remember**: Security is not optional. One vulnerability can compromise the entire platform. When in doubt, err on the side of caution. For ML models, also consider privacy, fairness, and model poisoning attacks.
