from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime

# Input schemas that match the middleware payload format

class MiddlewareJDInput(BaseModel):
    """JD structure as provided by middleware"""
    jd_id: str  # External ID from middleware
    title: str
    description: str
    skills: List[str] = []
    responsibilities: List[str] = []

class MiddlewareCVInput(BaseModel):
    """CV structure as provided by middleware"""
    cv_id: str  # External ID from middleware
    jd_id: str
    filename: str
    base64_content: str  # Note: middleware uses 'base64_content', not 'original_base64'

class InitialSessionPayload(BaseModel):
    """Complete initial session payload from middleware"""
    session_id: str
    job_descriptions: List[MiddlewareJDInput]
    cvs: List[MiddlewareCVInput]
    status: Literal["pending", "active", "processing", "completed", "failed"] = "active"

class IncrementalCVPayload(BaseModel):
    """Payload for adding CVs to existing session"""
    status: Literal["pending", "active", "processing", "completed", "failed"] = "active"
    cvs: List[MiddlewareCVInput]

# Session-specific schemas
class JDCompletionPayload(BaseModel):
    """Payload for completing/expiring a specific JD in a session"""
    session_id: str
    jd_id: str
    status: Literal["completed", "expired", "failed"]  # Only completion statuses allowed
    completed_at: Optional[datetime] = None
    reason: Optional[str] = None  # Optional reason for completion/expiration