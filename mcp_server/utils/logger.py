"""
Logging utility for MCP Server
Implements structured logging for tool calls with timestamps, payloads, and responses
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import threading

from ..config import LOG_LEVEL, LOG_FILE, TOOL_LOG_FILE


def setup_logging(level: str = LOG_LEVEL) -> logging.Logger:
    """
    Setup application logging with both console and file handlers.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("mcp_server")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not setup file logging: {e}")
    
    return logger


class ToolLogger:
    """
    Structured logger for tool calls.
    
    Logs each tool call with:
    - Tool name
    - Request payload
    - Response data
    - Timestamp
    - Source URLs (if applicable)
    - Duration
    - Success/Error status
    """
    
    def __init__(self, log_file: Path = TOOL_LOG_FILE):
        self.log_file = log_file
        self._lock = threading.Lock()
        self._logger = setup_logging()
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_tool_call(
        self,
        tool_name: str,
        request_payload: dict,
        response_data: Any,
        duration_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        source_urls: Optional[list[str]] = None,
        cache_hit: bool = False,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log a tool call with full context.
        
        Args:
            tool_name: Name of the tool (web.search, rag.search)
            request_payload: Input parameters sent to the tool
            response_data: Output from the tool
            duration_ms: Execution time in milliseconds
            success: Whether the call succeeded
            error_message: Error message if failed
            source_urls: List of source URLs accessed
            cache_hit: Whether result was from cache
            metadata: Additional metadata to log
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tool_name": tool_name,
            "request": self._sanitize_payload(request_payload),
            "response": self._summarize_response(response_data),
            "duration_ms": round(duration_ms, 2),
            "success": success,
            "cache_hit": cache_hit,
        }
        
        if error_message:
            log_entry["error"] = error_message
        
        if source_urls:
            log_entry["source_urls"] = source_urls
        
        if metadata:
            log_entry["metadata"] = metadata
        
        # Write to JSONL file
        with self._lock:
            try:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except IOError as e:
                self._logger.error(f"Failed to write tool log: {e}")
        
        # Also log to standard logger
        if success:
            self._logger.info(
                f"Tool call: {tool_name} | Duration: {duration_ms:.2f}ms | "
                f"Cache: {'HIT' if cache_hit else 'MISS'}"
            )
        else:
            self._logger.error(
                f"Tool call failed: {tool_name} | Error: {error_message}"
            )
    
    def _sanitize_payload(self, payload: dict) -> dict:
        """
        Remove sensitive information from payload before logging.
        """
        sensitive_keys = {"api_key", "password", "token", "secret", "auth"}
        sanitized = {}
        
        for key, value in payload.items():
            if any(sk in key.lower() for sk in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _summarize_response(self, response: Any) -> Any:
        """
        Summarize large responses to avoid log bloat.
        """
        if isinstance(response, list):
            if len(response) > 10:
                return {
                    "_type": "list",
                    "_count": len(response),
                    "_sample": response[:3],
                    "_truncated": True
                }
            return response
        
        if isinstance(response, dict):
            # Truncate large text fields
            summarized = {}
            for key, value in response.items():
                if isinstance(value, str) and len(value) > 500:
                    summarized[key] = value[:500] + "...[truncated]"
                else:
                    summarized[key] = value
            return summarized
        
        return response
    
    def get_recent_logs(self, limit: int = 100) -> list[dict]:
        """
        Retrieve recent tool call logs.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of log entries (most recent first)
        """
        logs = []
        
        try:
            if self.log_file.exists():
                with open(self.log_file, "r") as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except IOError:
            pass
        
        # Return most recent entries
        return logs[-limit:][::-1]
    
    def get_stats(self) -> dict:
        """
        Get statistics about tool calls.
        """
        logs = self.get_recent_logs(limit=10000)
        
        if not logs:
            return {"total_calls": 0}
        
        web_calls = [l for l in logs if l.get("tool_name") == "web.search"]
        rag_calls = [l for l in logs if l.get("tool_name") == "rag.search"]
        
        def calc_stats(calls: list) -> dict:
            if not calls:
                return {"count": 0}
            
            durations = [c.get("duration_ms", 0) for c in calls]
            successes = sum(1 for c in calls if c.get("success", False))
            cache_hits = sum(1 for c in calls if c.get("cache_hit", False))
            
            return {
                "count": len(calls),
                "success_rate": round(successes / len(calls) * 100, 1),
                "cache_hit_rate": round(cache_hits / len(calls) * 100, 1),
                "avg_duration_ms": round(sum(durations) / len(durations), 2),
                "min_duration_ms": round(min(durations), 2),
                "max_duration_ms": round(max(durations), 2),
            }
        
        return {
            "total_calls": len(logs),
            "web_search": calc_stats(web_calls),
            "rag_search": calc_stats(rag_calls),
        }


# Global tool logger instance
tool_logger = ToolLogger()

