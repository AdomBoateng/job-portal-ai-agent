from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import cvs, sessions, jds, reports, middleware, ai_settings

# Import logging and middleware
from app.utils.logging_config import configure_for_environment, get_logger
from app.middleware.error_handlers import (
    ExceptionHandlerMiddleware, 
    RequestLoggingMiddleware, 
    PerformanceMiddleware,
    HealthCheckMiddleware
)

# Configure logging first
configure_for_environment()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager"""
    # Startup
    logger.info("AI Agent API starting up...")
    logger.info("Initializing database indexes...")
    
    try:
        from app.services.db import init_indexes
        await init_indexes()
        logger.info("Database indexes initialized successfully")
    except Exception as e:
        logger.warning(f"Database index initialization had issues: {e}")
        logger.info("Application will continue - some operations may be slower without indexes")
    
    logger.info("AI Agent API startup completed")
    
    yield
    
    # Shutdown
    logger.info("AI Agent API shutting down...")
    logger.info("AI Agent API shutdown completed")

app = FastAPI(title="AI Agent API", version="1.0.0", lifespan=lifespan)

# Add middleware in order (LIFO - Last In, First Out)
# Exception handler should be the outermost middleware
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(PerformanceMiddleware, slow_request_threshold=2.0)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(HealthCheckMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - handles both GET and HEAD requests for health checks"""
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to the AI Agent API", "version": "1.0.0", "status": "ok"}

@app.get("/health")
@app.head("/health")
async def health_check():
    """Health check endpoint - handles both GET and HEAD requests"""
    return {"status": "healthy", "timestamp": "2025-10-13T10:00:00Z"}

# Include routers
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(jds.router, prefix="/api/jds", tags=["jds"])
app.include_router(cvs.router, prefix="/api/cvs", tags=["cvs"])
app.include_router(ai_settings.router, prefix="/api", tags=["ai-settings"])
app.include_router(reports.router,prefix="/api/match", tags=["reports"])  # reports has prefix="/sessions"
app.include_router(middleware.router)  # middleware has prefix="/middleware"

logger.info("AI Agent API initialized successfully")

