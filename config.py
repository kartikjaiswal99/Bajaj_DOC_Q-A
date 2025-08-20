# ==============================================================================
#                    CONFIGURATION AND ENVIRONMENT MANAGEMENT
# ==============================================================================
#
# This module handles all environment variables and configuration settings
# for the Advanced Document Q&A System. It provides centralized configuration
# management with validation and default values.
#
# CONFIGURATION CATEGORIES:
# =========================
# 1. Required Settings: Critical environment variables that must be set
# 2. Optional Settings: Configurable parameters with sensible defaults
# 3. Performance Settings: Tunable parameters for optimization
# 4. Security Settings: Authentication and access control parameters
#
# ENVIRONMENT VARIABLES:
# ======================
# Required:
# - OPENAI_API_KEY: OpenAI API key for embeddings and chat completions
#
# Optional:
# - OPENAI_MODEL: Chat model to use (default: gpt-4o-mini)
# - EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
# - MAX_WORKERS: Maximum parallel workers (default: 4)
# - CHUNK_SIZE: Text chunk size for processing (default: 1500)
# - DEBUG: Enable debug mode (default: false)
#
# USAGE:
# ======
# All configuration is loaded automatically when this module is imported.
# Environment variables are validated at startup to catch configuration
# errors early in the application lifecycle.
#
# ==============================================================================

import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
# This should be called before any other imports that depend on env vars
load_dotenv()

# ==============================================================================
#                           REQUIRED CONFIGURATION
# ==============================================================================

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required environment variables
if not OPENAI_API_KEY:
    raise ValueError(
        "❌ OPENAI_API_KEY is required but not found in environment variables.\n"
        "Please set your OpenAI API key in one of the following ways:\n"
        "1. Create a .env file with: OPENAI_API_KEY=your_api_key_here\n"
        "2. Set environment variable: export OPENAI_API_KEY=your_api_key_here\n"
        "3. Add to your shell profile for persistence\n\n"
        "Get your API key from: https://platform.openai.com/api-keys"
    )

# ==============================================================================
#                           OPTIONAL CONFIGURATION
# ==============================================================================

# OpenAI Model Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")  # Optional organization ID

# Application Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes", "on")
APP_RELOAD = os.getenv("APP_RELOAD", "false").lower() in ("true", "1", "yes", "on")

# Performance Configuration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
VECTOR_STORE_K = int(os.getenv("VECTOR_STORE_K", "8"))
MAX_TOKENS_PER_BATCH = int(os.getenv("MAX_TOKENS_PER_BATCH", "200000"))

# Document Processing Configuration
MAX_DOCUMENT_SIZE = os.getenv("MAX_DOCUMENT_SIZE", "50MB")
SUPPORTED_FORMATS = os.getenv("SUPPORTED_FORMATS", "pdf,docx,eml").split(",")
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))

# Caching Configuration
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "10"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Time to live in seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# ==============================================================================
#                           ADVANCED CONFIGURATION
# ==============================================================================

# Vector Database Configuration (for future Pinecone integration)
PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME: Optional[str] = os.getenv("PINECONE_INDEX_NAME", "bajaj-doc-qa")

# Redis Configuration (for enhanced caching)
REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory")  # "memory" or "redis"

# Security Configuration
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")

# ==============================================================================
#                         CONFIGURATION VALIDATION
# ==============================================================================

def validate_configuration() -> bool:
    """
    Validate all configuration settings and return True if valid.
    
    This function performs comprehensive validation of all configuration
    parameters to ensure the application can start successfully.
    
    Returns:
        bool: True if all configuration is valid, raises ValueError otherwise
        
    Raises:
        ValueError: If any required configuration is missing or invalid
    """
    errors = []
    
    # Validate OpenAI configuration
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required")
    elif not OPENAI_API_KEY.startswith(("sk-", "sk_")):
        errors.append("OPENAI_API_KEY appears to be invalid (should start with 'sk-')")
    
    # Validate numeric configurations
    try:
        if MAX_WORKERS <= 0:
            errors.append("MAX_WORKERS must be greater than 0")
        if CHUNK_SIZE <= 100:
            errors.append("CHUNK_SIZE must be greater than 100")
        if VECTOR_STORE_K <= 0:
            errors.append("VECTOR_STORE_K must be greater than 0")
        if APP_PORT <= 0 or APP_PORT > 65535:
            errors.append("APP_PORT must be between 1 and 65535")
    except (ValueError, TypeError) as e:
        errors.append(f"Invalid numeric configuration: {e}")
    
    # Validate supported formats
    valid_formats = {"pdf", "docx", "eml", "txt"}
    for fmt in SUPPORTED_FORMATS:
        if fmt.lower() not in valid_formats:
            errors.append(f"Unsupported format: {fmt}")
    
    # Validate log level
    valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if LOG_LEVEL not in valid_log_levels:
        errors.append(f"Invalid LOG_LEVEL: {LOG_LEVEL}")
    
    if errors:
        error_message = "❌ Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_message)
    
    return True

def get_config_summary() -> dict:
    """
    Get a summary of current configuration settings.
    
    Returns:
        dict: Summary of configuration settings (sensitive values masked)
    """
    return {
        "openai": {
            "api_key_set": bool(OPENAI_API_KEY),
            "model": OPENAI_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "org_id_set": bool(OPENAI_ORG_ID)
        },
        "application": {
            "host": APP_HOST,
            "port": APP_PORT,
            "debug": DEBUG,
            "reload": APP_RELOAD
        },
        "performance": {
            "max_workers": MAX_WORKERS,
            "chunk_size": CHUNK_SIZE,
            "vector_store_k": VECTOR_STORE_K,
            "max_tokens_per_batch": MAX_TOKENS_PER_BATCH
        },
        "processing": {
            "max_document_size": MAX_DOCUMENT_SIZE,
            "supported_formats": SUPPORTED_FORMATS,
            "download_timeout": DOWNLOAD_TIMEOUT
        },
        "caching": {
            "max_size": CACHE_MAX_SIZE,
            "ttl": CACHE_TTL,
            "backend": CACHE_BACKEND
        },
        "logging": {
            "level": LOG_LEVEL,
            "format": LOG_FORMAT,
            "file": LOG_FILE
        }
    }

# ==============================================================================
#                           STARTUP VALIDATION
# ==============================================================================

# Validate configuration on import
try:
    validate_configuration()
    if DEBUG:
        print("✅ Configuration validation passed")
        import json
        print("📋 Configuration summary:")
        print(json.dumps(get_config_summary(), indent=2))
except ValueError as e:
    print(f"{e}")
    print("\n💡 For help with configuration, see SETUP.md")
    raise

# ==============================================================================
#                              UTILITIES
# ==============================================================================

def is_development() -> bool:
    """Check if application is running in development mode."""
    return DEBUG or APP_RELOAD

def is_production() -> bool:
    """Check if application is running in production mode."""
    return not is_development()

def get_openai_config() -> dict:
    """Get OpenAI configuration for API clients."""
    config = {
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }
    if OPENAI_ORG_ID:
        config["organization"] = OPENAI_ORG_ID
    return config

# Export commonly used configuration
__all__ = [
    "OPENAI_API_KEY",
    "OPENAI_MODEL", 
    "EMBEDDING_MODEL",
    "MAX_WORKERS",
    "CHUNK_SIZE",
    "DEBUG",
    "validate_configuration",
    "get_config_summary",
    "is_development",
    "is_production",
    "get_openai_config"
]