# Setup and Configuration Guide

This guide provides detailed instructions for setting up, configuring, and deploying the Advanced Document Q&A System.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Environment Configuration](#environment-configuration)
- [Database Setup](#database-setup)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting Setup](#troubleshooting-setup)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **OS**: Linux, macOS, or Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for API calls and document downloads

#### Recommended Requirements
- **OS**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: 3.10 or higher
- **Memory**: 16GB RAM for large documents
- **Storage**: 10GB free space
- **CPU**: Multi-core processor for parallel processing
- **Network**: High-speed internet for optimal performance

### Required Accounts and API Keys

#### OpenAI Account
1. **Create Account**: Visit [OpenAI Platform](https://platform.openai.com/)
2. **Generate API Key**: 
   - Go to API Keys section
   - Click "Create new secret key"
   - Copy and securely store the key
3. **Set Up Billing**: Add payment method for API usage
4. **Check Limits**: Verify rate limits and quotas

#### Optional Services
- **Pinecone**: For production vector database
- **MongoDB**: For document metadata storage
- **Redis**: For enhanced caching

## Installation Methods

### Method 1: Direct Installation with pip

#### Step 1: Clone Repository
```bash
git clone https://github.com/kartikjaiswal99/Bajaj_DOC_Q-A.git
cd Bajaj_DOC_Q-A
```

#### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install fastapi uvicorn langchain langchain-openai langchain-community \
            faiss-cpu pymupdf python-docx pydantic python-dotenv requests

# Or install all at once
pip install -r requirements.txt  # If requirements.txt exists

# Or using pyproject.toml
pip install -e .
```

### Method 2: Poetry Installation

#### Step 1: Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Step 2: Install Project
```bash
git clone https://github.com/kartikjaiswal99/Bajaj_DOC_Q-A.git
cd Bajaj_DOC_Q-A
poetry install
poetry shell
```

### Method 3: Conda Installation

#### Step 1: Create Conda Environment
```bash
conda create -n bajaj-doc-qa python=3.10
conda activate bajaj-doc-qa
```

#### Step 2: Install Dependencies
```bash
# Install from conda-forge
conda install -c conda-forge fastapi uvicorn

# Install remaining with pip
pip install langchain langchain-openai langchain-community \
            faiss-cpu pymupdf python-docx pydantic python-dotenv
```

### Method 4: Development Installation

#### Step 1: Install Development Dependencies
```bash
pip install fastapi[all] uvicorn[standard] pytest pytest-asyncio \
            black flake8 mypy pre-commit
```

#### Step 2: Install Pre-commit Hooks
```bash
pre-commit install
```

## Environment Configuration

### Step 1: Create Environment File

Create a `.env` file in the project root:

```bash
touch .env
```

### Step 2: Configure Environment Variables

Add the following variables to your `.env` file:

```env
# Required Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional OpenAI Configuration
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_ORG_ID=your_org_id  # If using organization account

# Application Configuration
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
DEBUG=false

# Performance Configuration
MAX_WORKERS=4
CHUNK_SIZE=1500
VECTOR_STORE_K=8
MAX_TOKENS_PER_BATCH=200000

# Document Processing Configuration
MAX_DOCUMENT_SIZE=50MB
SUPPORTED_FORMATS=pdf,docx,eml
DOWNLOAD_TIMEOUT=30

# Caching Configuration
CACHE_MAX_SIZE=10
CACHE_TTL=3600

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=app.log
```

### Step 3: Validate Configuration

Create a script to validate your configuration:

```python
# validate_config.py
import os
from dotenv import load_dotenv

def validate_config():
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    validate_config()
```

Run validation:
```bash
python validate_config.py
```

### Environment Variable Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | None | ✅ Yes |
| `OPENAI_MODEL` | Chat model to use | `gpt-4o-mini` | ❌ No |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` | ❌ No |
| `APP_HOST` | Server host | `0.0.0.0` | ❌ No |
| `APP_PORT` | Server port | `8000` | ❌ No |
| `MAX_WORKERS` | Parallel workers | `4` | ❌ No |
| `CHUNK_SIZE` | Text chunk size | `1500` | ❌ No |
| `DEBUG` | Debug mode | `false` | ❌ No |

## Database Setup

### FAISS Vector Store (Default)

FAISS is used by default and requires no additional setup. It creates in-memory vector indices.

#### Advantages:
- No external dependencies
- Fast for development
- Simple setup

#### Limitations:
- Not persistent
- Single-node only
- Memory-bound

### Pinecone Vector Store (Production)

#### Step 1: Create Pinecone Account
1. Sign up at [Pinecone](https://www.pinecone.io/)
2. Create a new project
3. Get API key and environment

#### Step 2: Install Pinecone
```bash
pip install pinecone-client
```

#### Step 3: Configure Pinecone
```env
# Add to .env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=bajaj-doc-qa
```

#### Step 4: Initialize Pinecone Index
```python
# scripts/setup_pinecone.py
import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Create index
pinecone.create_index(
    name="bajaj-doc-qa",
    dimension=1536,  # OpenAI embedding dimension
    metric="cosine"
)
```

### Redis Cache (Optional)

#### Step 1: Install Redis
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

#### Step 2: Install Redis Python Client
```bash
pip install redis
```

#### Step 3: Configure Redis
```env
# Add to .env
REDIS_URL=redis://localhost:6379/0
CACHE_BACKEND=redis
```

## Development Setup

### Step 1: Install Development Tools

```bash
# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy pre-commit

# Install testing utilities
pip install httpx  # For testing FastAPI
```

### Step 2: Configure Development Environment

Create `dev.env` for development-specific settings:

```env
# Development settings
DEBUG=true
LOG_LEVEL=DEBUG
APP_RELOAD=true
OPENAI_MODEL=gpt-3.5-turbo  # Cheaper for development
```

### Step 3: Set Up Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
```

Install hooks:
```bash
pre-commit install
```

### Step 4: Create Development Scripts

#### Start Development Server
```bash
# scripts/dev_server.py
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv("dev.env")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=True,
        log_level="debug"
    )
```

#### Run Tests
```bash
# scripts/run_tests.py
import subprocess
import sys

def run_tests():
    commands = [
        ["python", "-m", "pytest", "tests/", "-v"],
        ["python", "-m", "black", "--check", "."],
        ["python", "-m", "flake8", "."],
        ["python", "-m", "mypy", "."]
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

if __name__ == "__main__":
    run_tests()
```

### Step 5: Create Test Configuration

#### Create Test Database
```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_document_url():
    return "https://example.com/test-document.pdf"

@pytest.fixture
def sample_questions():
    return [
        "What is this document about?",
        "Who is the target audience?"
    ]
```

#### Sample Test
```python
# tests/test_api.py
def test_hackrx_run_endpoint(client, sample_document_url, sample_questions):
    response = client.post(
        "/hackrx/run",
        json={
            "documents": sample_document_url,
            "questions": sample_questions
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert len(data["answers"]) == len(sample_questions)
```

## Production Deployment

### Step 1: Prepare Production Environment

#### Create Production Configuration
```env
# prod.env
DEBUG=false
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false

# Production OpenAI settings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Performance settings
MAX_WORKERS=8
CHUNK_SIZE=2000
VECTOR_STORE_K=10

# Security settings
ALLOWED_HOSTS=your-domain.com
CORS_ORIGINS=https://your-frontend.com
```

### Step 2: Install Production Server

```bash
# Install production WSGI server
pip install gunicorn

# Or install with extra features
pip install gunicorn[setproctitle]
```

### Step 3: Create Production Startup Script

```python
# gunicorn_config.py
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.getenv('APP_PORT', 8000)}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
keepalive = 2

# Restart workers
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "/var/log/bajaj-doc-qa/access.log"
errorlog = "/var/log/bajaj-doc-qa/error.log"
loglevel = "info"

# Process naming
proc_name = "bajaj-doc-qa"

# Security
limit_request_line = 0
limit_request_fields = 100
limit_request_field_size = 8190
```

### Step 4: Create Systemd Service

```ini
# /etc/systemd/system/bajaj-doc-qa.service
[Unit]
Description=Bajaj Document Q&A API
After=network.target

[Service]
Type=notify
User=app
Group=app
WorkingDirectory=/opt/bajaj-doc-qa
Environment=PATH=/opt/bajaj-doc-qa/venv/bin
EnvironmentFile=/opt/bajaj-doc-qa/prod.env
ExecStart=/opt/bajaj-doc-qa/venv/bin/gunicorn -c gunicorn_config.py app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
sudo systemctl enable bajaj-doc-qa
sudo systemctl start bajaj-doc-qa
sudo systemctl status bajaj-doc-qa
```

### Step 5: Configure Reverse Proxy

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/bajaj-doc-qa
server {
    listen 80;
    server_name your-domain.com;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
    
    location / {
        limit_req zone=api burst=5 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for large document processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/;
        access_log off;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/bajaj-doc-qa /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Docker Deployment

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Change ownership to app user
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run application
CMD ["python", "main_new.py"]
```

### Step 2: Create Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=false
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    depends_on:
      - redis
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - app
    restart: unless-stopped

volumes:
  redis_data:
```

### Step 3: Build and Deploy

```bash
# Build image
docker build -t bajaj-doc-qa .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3
```

### Step 4: Docker Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    image: bajaj-doc-qa:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "0.5"
          memory: 1G
        reservations:
          cpus: "0.25"
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DEBUG=false
      - LOG_LEVEL=INFO
    networks:
      - app-network
    
networks:
  app-network:
    driver: overlay
```

## Monitoring and Logging

### Step 1: Configure Logging

```python
# logging_config.py
import logging
import sys
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

def setup_logging():
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
```

### Step 2: Application Metrics

```python
# metrics.py
import time
import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage')

def start_metrics_server():
    start_http_server(9090)

def update_system_metrics():
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    CPU_USAGE.set(psutil.cpu_percent())
```

### Step 3: Health Checks

```python
# health.py
from fastapi import APIRouter
import psutil
import openai

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@router.get("/health/detailed")
async def detailed_health_check():
    # Check system resources
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent()
    disk_percent = psutil.disk_usage('/').percent
    
    # Check external services
    openai_status = await check_openai_connection()
    
    return {
        "status": "healthy" if all([
            memory_percent < 90,
            cpu_percent < 90,
            disk_percent < 90,
            openai_status
        ]) else "unhealthy",
        "checks": {
            "memory": {"percent": memory_percent, "status": "ok" if memory_percent < 90 else "warning"},
            "cpu": {"percent": cpu_percent, "status": "ok" if cpu_percent < 90 else "warning"},
            "disk": {"percent": disk_percent, "status": "ok" if disk_percent < 90 else "warning"},
            "openai": {"status": "ok" if openai_status else "error"}
        }
    }

async def check_openai_connection():
    try:
        # Simple API test
        response = await openai.Model.list()
        return True
    except:
        return False
```

## Troubleshooting Setup

### Common Installation Issues

#### Issue 1: Python Version Compatibility
```bash
# Check Python version
python --version

# If version is too old, install newer Python
sudo apt update
sudo apt install python3.10
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

#### Issue 2: FAISS Installation Fails
```bash
# Install system dependencies
sudo apt install libopenblas-dev

# Try CPU version specifically
pip install faiss-cpu==1.7.4

# For Apple Silicon Macs
conda install -c conda-forge faiss-cpu
```

#### Issue 3: PyMuPDF Installation Issues
```bash
# Install system dependencies
sudo apt install libmupdf-dev mupdf-tools

# Try alternative installation
pip install --upgrade pip
pip install pymupdf==1.23.8
```

### Runtime Issues

#### Issue 1: OpenAI API Errors
```python
# Test OpenAI connection
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    models = openai.Model.list()
    print("✅ OpenAI connection successful")
except Exception as e:
    print(f"❌ OpenAI connection failed: {e}")
```

#### Issue 2: Memory Issues
```bash
# Monitor memory usage
free -h
htop

# Adjust worker count
export MAX_WORKERS=2

# Use smaller chunk sizes
export CHUNK_SIZE=1000
```

#### Issue 3: Port Already in Use
```bash
# Find process using port
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>

# Use different port
export APP_PORT=8080
```

### Performance Issues

#### Issue 1: Slow Document Processing
```python
# Check document size
import requests
response = requests.head(document_url)
size = int(response.headers.get('content-length', 0))
print(f"Document size: {size / (1024*1024):.2f} MB")

# Optimize chunking
export CHUNK_SIZE=2000
export MAX_TOKENS_PER_BATCH=100000
```

#### Issue 2: OpenAI Rate Limits
```python
# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_openai_api():
    # Your OpenAI API call
    pass
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG
export PYTHONPATH=.

# Run with verbose logging
python -u main_new.py
```

### Support and Help

#### Log Analysis
```bash
# Search for errors
grep -i error logs/app.log

# Monitor real-time logs
tail -f logs/app.log

# Check system logs
journalctl -u bajaj-doc-qa -f
```

#### Performance Profiling
```python
# Profile API endpoint
import cProfile
import pstats

def profile_endpoint():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your API call here
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

For additional help, check the [Troubleshooting section in README.md](./README.md#troubleshooting) or create an issue in the repository.

---

This setup guide covers all aspects of installation, configuration, and deployment. Follow the appropriate sections based on your deployment environment and requirements.