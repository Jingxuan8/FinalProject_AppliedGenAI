"""
Configuration for MCP Server
All sensitive values should be set via environment variables
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# LLM Configuration (model-agnostic)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, google, etc.
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# Web Search API Configuration
WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "serper")  # serper, brave, bing
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY", os.getenv("SERPER_API_KEY", ""))
WEB_SEARCH_BASE_URL = os.getenv("WEB_SEARCH_BASE_URL", "https://google.serper.dev/search")

# Cache Configuration
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "180"))  # 60-300s as per requirement
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Vector Store Configuration (for RAG)
# BASE_DIR is now the project root (mcp_server/ -> project root)
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "data" / "vector_store"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "games_accessories")

# MCP Server Configuration
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8765"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # stdio or http

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "mcp_server.log"
TOOL_LOG_FILE = LOGS_DIR / "tool_calls.jsonl"

# Safety Configuration
ALLOWED_DOMAINS = [
    "amazon.com",
    "google.com",
    "bing.com",
    "walmart.com",
    "target.com",
]
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_RPM", "30"))

