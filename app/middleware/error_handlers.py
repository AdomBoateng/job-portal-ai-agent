"""
Global Exception Handler Middleware for AI Agent API
"""
import traceback
import uuid
from datetime import datetime
from typing import Any

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import ValidationError

from app.utils.exceptions import AIAgentBaseException, map_to_http_exception
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Global exception handler middleware"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to all log messages
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": request.client.host if request.client else "unknown"
            }
        )
        
        try:
            response = await call_next(request)
            
            # Log successful response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except AIAgentBaseException as exc:
            # Handle custom exceptions
            logger.error(
                f"Custom exception in {request.method} {request.url.path}: {exc.message}",
                extra={
                    "request_id": request_id,
                    "exception_type": exc.__class__.__name__,
                    "error_code": exc.error_code,
                    "details": exc.details,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            http_exc = map_to_http_exception(exc)
            return await self._create_error_response(request_id, http_exc.status_code, http_exc.detail)
            
        except RequestValidationError as exc:
            # Handle FastAPI validation errors (400 Bad Request)
            logger.error(
                f"Validation error in {request.method} {request.url.path}: {exc}",
                extra={
                    "request_id": request_id,
                    "validation_errors": exc.errors(),
                    "method": request.method,
                    "path": request.url.path,
                    "body": exc.body if hasattr(exc, 'body') else "N/A"
                }
            )
            
            # Format validation errors for client
            validation_details = {
                "error": "Validation failed",
                "message": "Request data validation failed",
                "validation_errors": exc.errors(),
                "request_id": request_id
            }
            
            return await self._create_error_response(request_id, 422, validation_details)
            
        except ValidationError as exc:
            # Handle Pydantic validation errors
            logger.error(
                f"Pydantic validation error in {request.method} {request.url.path}: {exc}",
                extra={
                    "request_id": request_id,
                    "validation_errors": exc.errors(),
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            validation_details = {
                "error": "Data validation failed",
                "message": "Invalid data format or values",
                "validation_errors": exc.errors(),
                "request_id": request_id
            }
            
            return await self._create_error_response(request_id, 400, validation_details)
            
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            logger.warning(
                f"HTTP exception in {request.method} {request.url.path}: {exc.detail}",
                extra={
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "method": request.method,
                    "path": request.url.path
                }
            )
            
            return await self._create_error_response(request_id, exc.status_code, exc.detail)
            
        except Exception as exc:
            # Handle unexpected exceptions
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {str(exc)}",
                extra={
                    "request_id": request_id,
                    "exception_type": exc.__class__.__name__,
                    "traceback": traceback.format_exc(),
                    "method": request.method,
                    "path": request.url.path
                },
                exc_info=True
            )
            
            # Don't expose internal errors in production
            error_detail = {
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "request_id": request_id
            }
            
            return await self._create_error_response(request_id, 500, error_detail)
    
    async def _create_error_response(self, request_id: str, status_code: int, detail: Any) -> JSONResponse:
        """Create standardized error response"""
        
        # Ensure detail is a dictionary
        if isinstance(detail, str):
            detail = {"message": detail}
        elif not isinstance(detail, dict):
            detail = {"message": str(detail)}
        
        # Add standard fields
        error_response = {
            "success": False,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "status_code": status_code,
            **detail
        }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response,
            headers={"X-Request-ID": request_id}
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging"""
    
    async def dispatch(self, request: Request, call_next):
        import time
        
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log request details
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body for logging (be careful with large files)
                body = await request.body()
                if len(body) < 10000:  # Only log small bodies
                    request_body = body.decode('utf-8', errors='ignore')[:1000]  # Truncate long bodies
                else:
                    request_body = f"<Large body: {len(body)} bytes>"
            except Exception:
                request_body = "<Unable to read body>"
        
        logger.debug(
            f"Request details: {request.method} {request.url}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request_body,
                "client_ip": request.client.host if request.client else "unknown"
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            # Log response details
            logger.info(
                f"Response: {request.method} {request.url.path} - {response.status_code} in {processing_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "response_headers": dict(response.headers)
                }
            )
            
            return response
            
        except Exception as exc:
            processing_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} after {processing_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "exception": str(exc)
                }
            )
            raise


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring"""
    
    def __init__(self, app, slow_request_threshold: float = 2.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next):
        import time
        
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        response = await call_next(request)
        
        processing_time = time.time() - start_time
        
        # Log performance metrics
        if processing_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} took {processing_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "threshold": self.slow_request_threshold,
                    "method": request.method,
                    "path": request.url.path
                }
            )
        else:
            logger.debug(
                f"Request performance: {request.method} {request.url.path} - {processing_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "processing_time": processing_time
                }
            )
        
        # Add performance headers
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        
        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware to handle health check requests without logging"""
    
    HEALTH_PATHS = ["/health", "/healthz", "/ping", "/status"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip detailed logging for health check endpoints
        if request.url.path in self.HEALTH_PATHS:
            return await call_next(request)
        
        return await call_next(request)