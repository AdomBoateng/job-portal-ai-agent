"""
Session Management Service for basic session operations and JD completion
"""
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from app.services.db import sessions_coll

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages basic session operations and JD completion logic"""
    
    # Session status constants
    STATUS_PENDING = "pending"
    STATUS_ACTIVE = "active"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_EXPIRED = "expired"
    STATUS_FAILED = "failed"
    
    @staticmethod
    async def complete_jd_in_session(session_id: str, jd_id: str, status: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Complete/expire a specific JD within a session"""
        from app.services.db import jds_coll
        
        # Verify session exists
        session = await sessions_coll.find_one({"session_id": session_id})
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Verify JD exists in this session
        jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
        if not jd:
            raise ValueError(f"JD {jd_id} not found in session {session_id}")
        
        # Update the JD with completion status
        jd_update_data = {
            "status": status,
            "completed_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if reason:
            jd_update_data["completion_reason"] = reason
        
        await jds_coll.update_one(
            {"jd_id": jd_id, "session_id": session_id},
            {"$set": jd_update_data}
        )
        
        # Check if all JDs in the session are completed
        all_jds = await jds_coll.find({"session_id": session_id}).to_list(length=None)
        completed_jds = [jd for jd in all_jds if jd.get("status") in [SessionManager.STATUS_COMPLETED, SessionManager.STATUS_EXPIRED, SessionManager.STATUS_FAILED]]
        
        session_status = session.get("status", SessionManager.STATUS_PENDING)
        
        if len(completed_jds) == len(all_jds) and len(all_jds) > 0:
            # All JDs are completed, mark session as completed
            session_status = SessionManager.STATUS_COMPLETED
            logger.info(f"All JDs completed in session {session_id}, marking session as completed")
        elif len(completed_jds) > 0:
            # Some JDs completed, but not all
            session_status = SessionManager.STATUS_ACTIVE
        
        # Update session status
        await sessions_coll.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": session_status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        logger.info(f"Completed JD {jd_id} in session {session_id} with status {status}")
        
        return {
            "session_id": session_id,
            "jd_id": jd_id,
            "jd_status": status,
            "session_status": session_status,
            "reason": reason,
            "completed_jds_count": len(completed_jds),
            "total_jds_count": len(all_jds),
            "message": f"JD {jd_id} marked as {status} in session {session_id}"
        }