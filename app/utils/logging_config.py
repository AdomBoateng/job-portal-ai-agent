"""
Centralized Logging Configuration for AI Agent API
"""
import logging
import logging.config
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = True,
    format_style: str = "detailed"
) -> None:
    """
    Setup centralized logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/ai_agent.log)
        enable_console: Enable console logging
        enable_file: Enable file logging
        format_style: Format style ('simple', 'detailed', 'json')
    """
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Default log file
    if log_file is None:
        log_file = log_dir / f"ai_agent_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Format configurations
    formats = {
        "simple": "%(levelname)s - %(name)s - %(message)s",
        "detailed": "%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s:%(lineno)-4d | %(message)s",
        "json": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
    }
    
    # Logging configuration dictionary
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": formats.get(format_style, formats["detailed"]),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": formats["simple"]
            }
        },
        "handlers": {},
        "loggers": {
            "": {  # Root logger
                "level": level,
                "handlers": [],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            }
        }
    }
    
    # Console handler
    if enable_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "simple" if format_style == "simple" else "detailed",
            "stream": "ext://sys.stdout"
        }
        config["loggers"][""]["handlers"].append("console")
        config["loggers"]["uvicorn"]["handlers"].append("console")
        config["loggers"]["uvicorn.access"]["handlers"].append("console")
    
    # File handler
    if enable_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
        config["loggers"][""]["handlers"].append("file")
        config["loggers"]["uvicorn"]["handlers"].append("file")
    
    # Error file handler for errors and above
    if enable_file:
        error_log_file = log_dir / f"ai_agent_errors_{datetime.now().strftime('%Y%m%d')}.log"
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(error_log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
        config["loggers"][""]["handlers"].append("error_file")
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log the configuration setup
    logger = logging.getLogger("ai_agent.logging")
    logger.info(f"Logging configured - Level: {level}, Console: {enable_console}, File: {enable_file}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent naming
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"ai_agent.{name}")


def log_function_call(func):
    """
    Decorator to log function calls with parameters and execution time
    """
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        # Log function entry
        logger.debug(f"Entering {func.__name__} with args={len(args)}, kwargs={list(kwargs.keys())}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)}")
            raise
    
    # Return appropriate wrapper based on function type
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_api_call(operation: str):
    """
    Decorator to log API endpoint calls
    """
    def decorator(func):
        import functools
        import time
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            logger = get_logger(f"api.{func.__module__}")
            start_time = time.time()
            
            # Extract request info if available
            request_info = {}
            for arg in args:
                if hasattr(arg, 'method') and hasattr(arg, 'url'):
                    request_info = {
                        "method": arg.method,
                        "url": str(arg.url),
                        "client": getattr(arg.client, 'host', 'unknown') if hasattr(arg, 'client') else 'unknown'
                    }
                    break
            
            logger.info(f"API {operation} started - {func.__name__}", extra=request_info)
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"API {operation} completed in {execution_time:.3f}s", extra={"execution_time": execution_time})
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"API {operation} failed after {execution_time:.3f}s: {str(e)}", 
                           extra={"execution_time": execution_time, "error": str(e)})
                raise
        
        return wrapper
    return decorator


# Environment-based configuration
def configure_for_environment():
    """Configure logging based on environment variables"""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    if environment == "production":
        setup_logging(
            level=log_level,
            enable_console=True,
            enable_file=True,
            format_style="detailed"
        )
    elif environment == "development":
        setup_logging(
            level="DEBUG",
            enable_console=True,
            enable_file=True,
            format_style="detailed"
        )
    elif environment == "testing":
        setup_logging(
            level="WARNING",
            enable_console=True,
            enable_file=False,
            format_style="simple"
        )
    else:
        # Default configuration
        setup_logging(level=log_level)


# Performance monitoring context manager
class PerformanceMonitor:
    """Context manager for monitoring performance with logging"""
    
    def __init__(self, operation_name: str, logger: logging.Logger = None, threshold_ms: float = 1000):
        self.operation_name = operation_name
        self.logger = logger or get_logger("performance")
        self.threshold_ms = threshold_ms
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (time.time() - self.start_time) * 1000  # Convert to milliseconds
        
        if exc_type is not None:
            self.logger.error(f"{self.operation_name} failed after {execution_time:.2f}ms: {exc_val}")
        elif execution_time > self.threshold_ms:
            self.logger.warning(f"{self.operation_name} completed in {execution_time:.2f}ms (exceeded threshold {self.threshold_ms}ms)")
        else:
            self.logger.info(f"{self.operation_name} completed in {execution_time:.2f}ms")


# Initialize logging on module import
if __name__ != "__main__":
    configure_for_environment()