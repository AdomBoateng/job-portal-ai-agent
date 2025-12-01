"""
Custom Exception Classes for AI Agent API
"""
from typing import Dict, Any
from fastapi import HTTPException


class AIAgentBaseException(Exception):
    """Base exception for AI Agent API"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.cause = cause
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/response"""
        result = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }
        if self.cause:
            result["cause"] = str(self.cause)
        return result


class ValidationError(AIAgentBaseException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        details = kwargs.get('details', {})
        if field:
            details['field'] = field
        if value is not None:
            details['invalid_value'] = str(value)
        super().__init__(message, error_code="VALIDATION_ERROR", details=details, **kwargs)


class DatabaseError(AIAgentBaseException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: str = None, collection: str = None, **kwargs):
        details = kwargs.get('details', {})
        if operation:
            details['operation'] = operation
        if collection:
            details['collection'] = collection
        super().__init__(message, error_code="DATABASE_ERROR", details=details, **kwargs)


class ModelError(AIAgentBaseException):
    """Raised when AI model operations fail"""
    
    def __init__(self, message: str, model_name: str = None, model_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if model_type:
            details['model_type'] = model_type
        super().__init__(message, error_code="MODEL_ERROR", details=details, **kwargs)


class ProcessingError(AIAgentBaseException):
    """Raised when CV/JD processing fails"""
    
    def __init__(self, message: str, document_id: str = None, document_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        if document_id:
            details['document_id'] = document_id
        if document_type:
            details['document_type'] = document_type
        super().__init__(message, error_code="PROCESSING_ERROR", details=details, **kwargs)


class ConfigurationError(AIAgentBaseException):
    """Raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None, **kwargs):
        details = kwargs.get('details', {})
        if config_key:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details, **kwargs)


class AuthenticationError(AIAgentBaseException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class AuthorizationError(AIAgentBaseException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Insufficient permissions", resource: str = None, **kwargs):
        details = kwargs.get('details', {})
        if resource:
            details['resource'] = resource
        super().__init__(message, error_code="AUTHORIZATION_ERROR", details=details, **kwargs)


class RateLimitError(AIAgentBaseException):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str, limit: int = None, window: str = None, **kwargs):
        details = kwargs.get('details', {})
        if limit:
            details['limit'] = limit
        if window:
            details['window'] = window
        super().__init__(message, error_code="RATE_LIMIT_ERROR", details=details, **kwargs)


class ExternalServiceError(AIAgentBaseException):
    """Raised when external service calls fail"""
    
    def __init__(self, message: str, service_name: str = None, status_code: int = None, **kwargs):
        details = kwargs.get('details', {})
        if service_name:
            details['service_name'] = service_name
        if status_code:
            details['status_code'] = status_code
        super().__init__(message, error_code="EXTERNAL_SERVICE_ERROR", details=details, **kwargs)


class BusinessLogicError(AIAgentBaseException):
    """Raised when business logic validation fails"""
    
    def __init__(self, message: str, rule: str = None, **kwargs):
        details = kwargs.get('details', {})
        if rule:
            details['business_rule'] = rule
        super().__init__(message, error_code="BUSINESS_LOGIC_ERROR", details=details, **kwargs)


# HTTP Exception Mapping
def map_to_http_exception(exc: AIAgentBaseException) -> HTTPException:
    """Map custom exceptions to HTTP exceptions"""
    
    status_code_mapping = {
        ValidationError: 400,
        ConfigurationError: 400,
        BusinessLogicError: 400,
        AuthenticationError: 401,
        AuthorizationError: 403,
        DatabaseError: 500,
        ModelError: 500,
        ProcessingError: 500,
        ExternalServiceError: 502,
        RateLimitError: 429
    }
    
    status_code = status_code_mapping.get(type(exc), 500)
    
    detail = {
        "error": exc.to_dict(),
        "message": exc.message
    }
    
    return HTTPException(status_code=status_code, detail=detail)


# Exception context manager for better error handling
class ExceptionContext:
    """Context manager for handling exceptions with additional context"""
    
    def __init__(self, operation: str, logger=None, **context):
        self.operation = operation
        self.logger = logger
        self.context = context
    
    def __enter__(self):
        if self.logger:
            self.logger.debug(f"Starting operation: {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.logger:
                self.logger.error(
                    f"Operation failed: {self.operation} - {exc_val}", 
                    extra={**self.context, "exception_type": exc_type.__name__}
                )
            
            # Re-raise custom exceptions as-is
            if isinstance(exc_val, AIAgentBaseException):
                return False
            
            # Wrap other exceptions
            if isinstance(exc_val, (KeyError, ValueError, TypeError)):
                wrapped_exc = ValidationError(
                    f"Validation error in {self.operation}: {str(exc_val)}",
                    details=self.context,
                    cause=exc_val
                )
                raise wrapped_exc from exc_val
            elif "database" in str(exc_val).lower() or "mongo" in str(exc_val).lower():
                wrapped_exc = DatabaseError(
                    f"Database error in {self.operation}: {str(exc_val)}",
                    operation=self.operation,
                    details=self.context,
                    cause=exc_val
                )
                raise wrapped_exc from exc_val
            else:
                # Wrap as generic processing error
                wrapped_exc = ProcessingError(
                    f"Processing error in {self.operation}: {str(exc_val)}",
                    details=self.context,
                    cause=exc_val
                )
                raise wrapped_exc from exc_val
        else:
            if self.logger:
                self.logger.debug(f"Operation completed: {self.operation}", extra=self.context)
        
        return False


# Retry decorator with exponential backoff
def retry_with_logging(
    max_attempts: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,),
    logger=None
):
    """Decorator to retry operations with exponential backoff and logging"""
    import time
    import functools
    from random import uniform
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}"
                        )
                    
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        sleep_time = backoff_factor * (2 ** attempt) + uniform(0, 1)
                        await asyncio.sleep(sleep_time)
                    else:
                        if logger:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise last_exception
            
            return None  # Should never reach here
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}"
                        )
                    
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        sleep_time = backoff_factor * (2 ** attempt) + uniform(0, 1)
                        time.sleep(sleep_time)
                    else:
                        if logger:
                            logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise last_exception
            
            return None  # Should never reach here
        
        import inspect
        import asyncio
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator