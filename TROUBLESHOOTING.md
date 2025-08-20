# Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues with the Advanced Document Q&A System.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Runtime Errors](#runtime-errors)
- [Performance Issues](#performance-issues)
- [API Errors](#api-errors)
- [Document Processing Issues](#document-processing-issues)
- [OpenAI API Issues](#openai-api-issues)
- [Debug Mode](#debug-mode)
- [Common Solutions](#common-solutions)

## Quick Diagnostics

### Health Check Script

Run this script to quickly diagnose common issues:

```python
#!/usr/bin/env python3
"""
Quick diagnostic script for the Document Q&A System.
Run this to identify common configuration and connectivity issues.
"""

import os
import sys
import requests
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"Python {version.major}.{version.minor} (need 3.8+)"
    return True, f"Python {version.major}.{version.minor}.{version.micro}"

def check_environment():
    """Check environment variables."""
    required_vars = ['OPENAI_API_KEY']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All required variables set"

def check_dependencies():
    """Check if key dependencies are installed."""
    required_packages = [
        'fastapi', 'uvicorn', 'langchain', 'openai', 
        'faiss', 'pymupdf', 'pydantic', 'python-dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All dependencies installed"

def check_api_server():
    """Check if API server is running."""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            return True, "API server is running"
        else:
            return False, f"API server returned HTTP {response.status_code}"
    except requests.ConnectionError:
        return False, "API server is not running"
    except Exception as e:
        return False, f"Error connecting to API: {e}"

def check_openai_connection():
    """Test OpenAI API connection."""
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "No OpenAI API key found"
        
        # Test with a simple API call
        client = openai.OpenAI(api_key=api_key)
        models = client.models.list()
        return True, "OpenAI API connection successful"
    except Exception as e:
        return False, f"OpenAI API error: {e}"

def main():
    """Run all diagnostic checks."""
    print("🔍 Document Q&A System Diagnostics")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment Variables", check_environment),
        ("Dependencies", check_dependencies),
        ("API Server", check_api_server),
        ("OpenAI Connection", check_openai_connection),
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        try:
            passed, message = check_func()
            status = "✅" if passed else "❌"
            print(f"{status} {name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}: Error running check - {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! System appears to be working correctly.")
    else:
        print("⚠️  Some issues detected. See solutions below.")
        print("\n💡 Quick fixes:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Set environment variables in .env file")
        print("   - Start API server: python main_new.py")
        print("   - Check OpenAI API key and billing status")

if __name__ == "__main__":
    main()
```

## Installation Issues

### Issue 1: Python Version Compatibility

**Error:**
```
SyntaxError: invalid syntax
```

**Solution:**
```bash
# Check Python version
python --version

# If version is < 3.8, install newer Python
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# macOS:
brew install python@3.10

# Windows:
# Download from python.org and install
```

### Issue 2: FAISS Installation Fails

**Error:**
```
ERROR: Failed building wheel for faiss-cpu
```

**Solution:**
```bash
# Install system dependencies first
# Ubuntu/Debian:
sudo apt install build-essential libopenblas-dev

# macOS:
brew install openblas

# Try alternative installation methods:
pip install --no-cache-dir faiss-cpu
# Or for Apple Silicon:
conda install -c conda-forge faiss-cpu
# Or specific version:
pip install faiss-cpu==1.7.4
```

### Issue 3: PyMuPDF Installation Issues

**Error:**
```
ERROR: Failed to build PyMuPDF
```

**Solution:**
```bash
# Install system dependencies
# Ubuntu/Debian:
sudo apt install libmupdf-dev mupdf-tools

# macOS:
brew install mupdf-tools

# Alternative installation:
pip install --upgrade pip setuptools wheel
pip install pymupdf==1.23.8
```

### Issue 4: Module Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:**
```bash
# Install all dependencies at once
pip install -r requirements.txt

# Or install individually if requirements.txt is missing
pip install fastapi uvicorn langchain langchain-openai langchain-community \
            faiss-cpu pymupdf python-docx pydantic python-dotenv requests

# Check virtual environment
which python
pip list | grep fastapi
```

## Configuration Problems

### Issue 1: OpenAI API Key Not Found

**Error:**
```
ValueError: Please set OPENAI_API_KEY in your .env file
```

**Solution:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env

# Or set environment variable
export OPENAI_API_KEY=your_api_key_here

# Verify it's set
python -c "import os; print('Key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

### Issue 2: Invalid API Key Format

**Error:**
```
openai.AuthenticationError: Invalid API key
```

**Solution:**
```bash
# Check API key format (should start with sk-)
echo $OPENAI_API_KEY | head -c 5

# Verify on OpenAI platform
# Visit: https://platform.openai.com/api-keys

# Test API key
python -c "
import openai
import os
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
try:
    models = client.models.list()
    print('✅ API key is valid')
except Exception as e:
    print(f'❌ API key error: {e}')
"
```

### Issue 3: Port Already in Use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find process using port 8000
sudo lsof -i :8000
sudo netstat -tulpn | grep :8000

# Kill the process
sudo kill -9 <PID>

# Or use different port
export APP_PORT=8080
python main_new.py

# Or in code:
uvicorn.run(app, host="0.0.0.0", port=8080)
```

## Runtime Errors

### Issue 1: Memory Errors with Large Documents

**Error:**
```
MemoryError: Unable to allocate memory
```

**Solution:**
```bash
# Check system memory
free -h

# Reduce batch sizes
export MAX_TOKENS_PER_BATCH=100000
export CHUNK_SIZE=1000
export MAX_WORKERS=2

# Monitor memory usage
top -p $(pgrep -f python)

# For very large documents, process in smaller sections
```

### Issue 2: Timeout Errors

**Error:**
```
TimeoutError: Request timed out
```

**Solution:**
```python
# Increase timeout in client code
import requests

response = requests.post(
    "http://localhost:8000/hackrx/run",
    json=payload,
    timeout=300  # 5 minutes
)

# For server-side, modify uvicorn settings
uvicorn.run(app, timeout_keep_alive=300)

# Check document size
import requests
response = requests.head(document_url)
size_mb = int(response.headers.get('content-length', 0)) / (1024*1024)
print(f"Document size: {size_mb:.2f} MB")
```

### Issue 3: CORS Errors in Browser

**Error:**
```
CORS policy: No 'Access-Control-Allow-Origin' header
```

**Solution:**
```python
# Add CORS middleware to app.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Performance Issues

### Issue 1: Slow Response Times

**Symptoms:**
- Requests taking >60 seconds
- High CPU usage
- Memory consumption growing

**Diagnosis:**
```python
import time
import psutil

def monitor_performance():
    start_time = time.time()
    
    # Monitor system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'available_memory_gb': memory.available / (1024**3)
    }
```

**Solutions:**
```bash
# Optimize configuration
export MAX_WORKERS=2  # Reduce for large documents
export CHUNK_SIZE=1500  # Optimize chunk size
export VECTOR_STORE_K=6  # Reduce retrieved documents

# Enable fast mode
export FAST_MODE=true

# Monitor and tune
htop  # Monitor CPU/memory in real-time
```

### Issue 2: OpenAI Rate Limiting

**Error:**
```
openai.RateLimitError: Rate limit exceeded
```

**Solution:**
```python
# Add exponential backoff
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_openai_with_retry():
    # Your OpenAI API call here
    pass

# Check rate limits
# Visit: https://platform.openai.com/account/rate-limits

# Upgrade OpenAI plan if needed
# Consider using gpt-3.5-turbo for development (cheaper)
export OPENAI_MODEL=gpt-3.5-turbo
```

## API Errors

### Issue 1: 422 Validation Error

**Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "questions"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Solution:**
```python
# Validate request format
import requests

# Correct format:
payload = {
    "documents": "https://example.com/doc.pdf",
    "questions": ["What is this about?"]
}

# Check payload before sending
print("Payload:", json.dumps(payload, indent=2))

response = requests.post(
    "http://localhost:8000/hackrx/run",
    json=payload  # Use json=, not data=
)
```

### Issue 2: 400 Document Processing Error

**Error:**
```json
{
  "detail": "Error: Could not process the provided document."
}
```

**Diagnosis:**
```python
# Test document accessibility
import requests

def test_document_url(url):
    try:
        response = requests.head(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")
        print(f"Content-Length: {response.headers.get('content-length')}")
        
        # Test actual download
        response = requests.get(url, timeout=30, stream=True)
        first_chunk = next(response.iter_content(chunk_size=1024))
        print(f"First 20 bytes: {first_chunk[:20]}")
        
        return True
    except Exception as e:
        print(f"Error accessing document: {e}")
        return False

# Test your document URL
test_document_url("https://example.com/document.pdf")
```

**Solutions:**
- Verify document URL is accessible
- Check document format (PDF, DOCX, EML only)
- Ensure document is not password-protected
- Try with smaller document for testing

## Document Processing Issues

### Issue 1: PDF Text Extraction Fails

**Error:**
```
Exception during PDF processing
```

**Diagnosis:**
```python
import fitz  # PyMuPDF

def diagnose_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        print(f"Page count: {len(doc)}")
        print(f"Metadata: {doc.metadata}")
        
        # Test first page
        page = doc[0]
        text = page.get_text()
        print(f"First page text length: {len(text)}")
        print(f"First 100 chars: {text[:100]}")
        
        doc.close()
        return True
    except Exception as e:
        print(f"PDF diagnosis error: {e}")
        return False
```

**Solutions:**
- Update PyMuPDF: `pip install --upgrade pymupdf`
- Try alternative PDF: Test with simple text-based PDF
- Check PDF is not corrupted: Open in PDF viewer
- For scanned PDFs: System doesn't support OCR (text-based PDFs only)

### Issue 2: DOCX Processing Fails

**Error:**
```
PackageNotFoundError: Package not found
```

**Solution:**
```python
# Test DOCX file
from docx import Document

def test_docx(file_path):
    try:
        doc = Document(file_path)
        print(f"Paragraph count: {len(doc.paragraphs)}")
        
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        
        print(f"Total text length: {len(text)}")
        return True
    except Exception as e:
        print(f"DOCX error: {e}")
        return False

# Update python-docx
pip install --upgrade python-docx
```

## Debug Mode

### Enable Comprehensive Debugging

```bash
# Set debug environment variables
export DEBUG=true
export LOG_LEVEL=DEBUG
export PYTHONPATH=.

# Run with verbose output
python -u main_new.py

# Or with uvicorn debug
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### Debug Configuration Script

```python
#!/usr/bin/env python3
"""Debug configuration and test basic functionality."""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_config():
    """Print debug configuration information."""
    print("🔧 Debug Configuration")
    print("=" * 30)
    
    # Environment variables
    env_vars = [
        'OPENAI_API_KEY', 'OPENAI_MODEL', 'EMBEDDING_MODEL',
        'MAX_WORKERS', 'CHUNK_SIZE', 'DEBUG'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        if 'API_KEY' in var and value != 'Not set':
            value = f"{value[:8]}...{value[-4:]}"  # Mask API key
        print(f"{var}: {value}")
    
    print("\n📁 File Structure")
    print("=" * 20)
    
    required_files = [
        'main_new.py', 'app.py', 'schemas.py', 
        'document_processor.py', 'rag_chain.py', 'config.py'
    ]
    
    for file in required_files:
        exists = "✅" if Path(file).exists() else "❌"
        print(f"{exists} {file}")

def test_imports():
    """Test importing all modules."""
    print("\n📦 Testing Imports")
    print("=" * 20)
    
    modules = [
        'fastapi', 'uvicorn', 'langchain', 'openai',
        'faiss', 'fitz', 'docx', 'pydantic'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

def test_basic_functionality():
    """Test basic system functionality."""
    print("\n🧪 Basic Functionality Test")
    print("=" * 30)
    
    try:
        # Test config loading
        from config import validate_configuration
        validate_configuration()
        print("✅ Configuration validation passed")
        
        # Test schema validation
        from schemas import HackathonInput
        test_input = HackathonInput(
            documents="https://example.com/test.pdf",
            questions=["Test question?"]
        )
        print("✅ Schema validation passed")
        
        # Test document processor import
        from document_processor import build_knowledge_base_from_urls
        print("✅ Document processor import successful")
        
        # Test RAG chain import
        from rag_chain import create_rag_chain
        print("✅ RAG chain import successful")
        
        print("\n🎉 All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Error in basic functionality: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config()
    test_imports()
    test_basic_functionality()
```

## Common Solutions

### 1. Complete System Reset

```bash
# Stop all processes
pkill -f "python.*main_new"
pkill -f uvicorn

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall dependencies
pip uninstall -y fastapi uvicorn langchain langchain-openai
pip install --no-cache-dir -r requirements.txt

# Reset configuration
rm -f .env
echo "OPENAI_API_KEY=your_key_here" > .env

# Test configuration
python -c "from config import validate_configuration; validate_configuration()"

# Start server
python main_new.py
```

### 2. Docker Reset (if using Docker)

```bash
# Stop and remove containers
docker-compose down -v

# Remove images
docker-compose down --rmi all

# Rebuild and start
docker-compose up --build
```

### 3. Virtual Environment Reset

```bash
# Remove existing virtual environment
rm -rf venv

# Create new virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Test installation
python -c "import fastapi, langchain, openai; print('✅ All imports successful')"
```

## Getting Help

### Enable Debug Logging

Add this to your Python code for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add debug prints
logger.debug("Processing document: %s", document_url)
logger.debug("Questions: %s", questions)
```

### Collect System Information

```bash
# System information script
cat > debug_info.sh << 'EOF'
#!/bin/bash
echo "=== System Debug Information ==="
echo "Date: $(date)"
echo "OS: $(uname -a)"
echo "Python: $(python --version 2>&1)"
echo "Pip: $(pip --version)"
echo "Memory: $(free -h | grep Mem)"
echo "Disk: $(df -h . | tail -1)"
echo "Process: $(ps aux | grep python | grep -v grep)"
echo "Network: $(netstat -tulpn | grep :8000 || echo 'Port 8000 not in use')"
echo "Environment:"
env | grep -E "(OPENAI|DEBUG|APP_|MAX_|CHUNK_)" | sort
EOF

chmod +x debug_info.sh
./debug_info.sh
```

For additional help:
1. Check the logs in `logs/app.log` (if configured)
2. Run the diagnostic script above
3. Check OpenAI API status: https://status.openai.com/
4. Verify your OpenAI billing and usage: https://platform.openai.com/usage

---

If you continue to experience issues after trying these solutions, please provide:
1. The output of the diagnostic script
2. Complete error messages and stack traces
3. Your system information (OS, Python version, etc.)
4. Steps to reproduce the issue