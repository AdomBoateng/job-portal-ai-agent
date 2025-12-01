# from ast import List
from fastapi import APIRouter, HTTPException, Request
from app.services.db import sessions_coll
from app.models.schemas import SessionModel
from typing import List

# Import logging and exceptions
from app.utils.logging_config import get_logger, PerformanceMonitor
from app.utils.exceptions import DatabaseError, ValidationError, ExceptionContext

# from datetime import datetime
# import uuid

router = APIRouter()
logger = get_logger(__name__)

@router.get("/all", response_model=List[SessionModel])
async def list_all_sessions(request: Request):
    """Get all sessions in the system"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info("Fetching all sessions", extra={"request_id": request_id})
    
    with PerformanceMonitor("list_all_sessions", logger):
        try:
            with ExceptionContext("fetch_sessions", logger, request_id=request_id):
                cursor = sessions_coll.find({})
                sessions = await cursor.to_list(length=None)
                
                logger.info(
                    f"Successfully fetched {len(sessions)} sessions", 
                    extra={"request_id": request_id, "session_count": len(sessions)}
                )
                
                return [SessionModel(**session) for session in sessions]
                
        except Exception as e:
            logger.error(
                f"Failed to fetch sessions: {str(e)}", 
                extra={"request_id": request_id}
            )
            raise DatabaseError(
                "Failed to retrieve sessions from database",
                operation="list_all_sessions",
                collection="sessions",
                cause=e
            )

@router.get("/{session_id}", response_model=SessionModel)
async def get_session(session_id: str, request: Request):
    """Fetch a session by ID"""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.info(
        f"Fetching session by ID: {session_id}", 
        extra={"request_id": request_id, "session_id": session_id}
    )
    
    if not session_id or not session_id.strip():
        logger.warning(
            "Invalid session ID provided", 
            extra={"request_id": request_id, "session_id": session_id}
        )
        raise ValidationError("Session ID cannot be empty", field="session_id", value=session_id)
    
    with PerformanceMonitor("get_session", logger):
        try:
            with ExceptionContext("fetch_session_by_id", logger, request_id=request_id, session_id=session_id):
                session = await sessions_coll.find_one({"session_id": session_id})
                
                if not session:
                    logger.warning(
                        f"Session not found: {session_id}", 
                        extra={"request_id": request_id, "session_id": session_id}
                    )
                    raise HTTPException(status_code=404, detail="Session not found")
                
                logger.info(
                    f"Successfully fetched session: {session_id}", 
                    extra={"request_id": request_id, "session_id": session_id}
                )
                
                return SessionModel(**session)
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(
                f"Failed to fetch session {session_id}: {str(e)}", 
                extra={"request_id": request_id, "session_id": session_id}
            )
            raise DatabaseError(
                f"Failed to retrieve session {session_id} from database",
                operation="get_session",
                collection="sessions",
                cause=e
            )