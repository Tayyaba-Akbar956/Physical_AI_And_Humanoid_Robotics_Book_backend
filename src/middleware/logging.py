import logging
import sys
from datetime import datetime, UTC
from pathlib import Path
import json
from typing import Dict, Any


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Set up application-wide logging configuration
    """
    # Create logger
    logger = logging.getLogger("rag_chatbot")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Also set level for uvicorn and fastapi loggers
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(getattr(logging, log_level.upper()))
    
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


def log_api_call(endpoint: str, method: str, duration: float, status_code: int, 
                user_id: str = None, session_id: str = None):
    """
    Log an API call with appropriate details
    """
    logger = logging.getLogger("rag_chatbot")
    logger.info(f"API_CALL: {method} {endpoint} | Status: {status_code} | Duration: {duration:.3f}s | "
                f"User: {user_id} | Session: {session_id}")


def log_error(error_id: str, error_type: str, message: str, details: Dict[str, Any] = None):
    """
    Log an error with appropriate details
    """
    logger = logging.getLogger("rag_chatbot")
    error_log = f"ERROR: ID={error_id} | Type={error_type} | Message={message}"
    if details:
        error_log += f" | Details={json.dumps(details)}"
    logger.error(error_log)


def log_user_interaction(user_id: str, session_id: str, interaction_type: str, 
                        content: str, module_context: str = None):
    """
    Log a user interaction for analytics
    """
    logger = logging.getLogger("rag_chatbot")
    interaction_log = (f"INTERACTION: User={user_id} | Session={session_id} | "
                       f"Type={interaction_type} | Module={module_context} | "
                       f"Content={content[:100]}...")  # Truncate long content
    logger.info(interaction_log)


def log_module_context_switch(user_id: str, session_id: str, old_module: str, new_module: str):
    """
    Log when a user switches module context
    """
    logger = logging.getLogger("rag_chatbot")
    switch_log = f"MODULE_SWITCH: User={user_id} | Session={session_id} | "
    switch_log += f"From={old_module} | To={new_module}"
    logger.info(switch_log)


def log_search_query(user_id: str, session_id: str, query: str, module_context: str, 
                    results_count: int, execution_time: float):
    """
    Log a search query for analytics
    """
    logger = logging.getLogger("rag_chatbot")
    search_log = (f"SEARCH_QUERY: User={user_id} | Session={session_id} | "
                  f"Module={module_context} | Query={query[:100]}... | "
                  f"Results={results_count} | Time={execution_time:.3f}s")
    logger.info(search_log)


class LoggingMiddleware:
    """
    FastAPI middleware for logging requests
    """
    def __init__(self, app):
        self.app = app
        setup_logging()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        start_time = datetime.now(UTC)

        # Get request details
        method = scope["method"]
        path = scope["path"]

        # Extract user and session IDs if available
        user_id = "unknown"
        session_id = "unknown"

        # Process the request
        async def send_wrapper(response):
            if response["type"] == "http.response.start":
                status_code = response["status"]

                # Calculate duration
                duration = (datetime.now(UTC) - start_time).total_seconds()

                # Log the API call
                log_api_call(
                    endpoint=path,
                    method=method,
                    duration=duration,
                    status_code=status_code,
                    user_id=user_id,
                    session_id=session_id
                )

            return await send(response)
        
        return await self.app(scope, receive, send_wrapper)


# Initialize logger
app_logger = setup_logging()

if __name__ == "__main__":
    # Example usage
    setup_logging(log_level="INFO", log_file="rag_chatbot.log")
    
    logger = logging.getLogger("rag_chatbot")
    logger.info("Logging setup complete")
    
    # Example logs
    log_api_call("/api/chat/query", "POST", 0.523, 200, "user123", "session456")
    log_user_interaction("user123", "session456", "question", "How do ROS nodes communicate?")
    log_search_query("user123", "session456", "ROS node communication", "module-2-ros2", 3, 0.482)