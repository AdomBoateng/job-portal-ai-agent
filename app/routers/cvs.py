#import base64
from fastapi import APIRouter, HTTPException, Query     #HTTPException, Query
from app.services.db import cvs_coll     #jds_coll, cvs_coll, sessions_coll
from app.models.schemas import CVModel
#from app.models.middleware_schemas import MiddlewareCVInput
#from datetime import datetime
from typing import List
# import uuid

router = APIRouter()

# def decode_base64_to_text(b64_string: str) -> str:
#     try:
#         decoded_bytes = base64.b64decode(b64_string)
#         return decoded_bytes.decode("utf-8", errors="ignore")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid base64 CV: {e}")

# @router.post("/", response_model=CVModel)
# async def add_cv(session_id: str, jd_id: str, cv: CVModel):
#     """Add CV to a JD within a session"""
#     # Ensure JD exists
#     jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
#     if not jd:
#         raise HTTPException(status_code=404, detail="JD not found for this session")

#     cv_id = str(uuid.uuid4())
#     parsed_text = None

#     if cv.original_base64:
#         parsed_text = decode_base64_to_text(cv.original_base64)

#     cv_data = cv.dict()
#     cv_data.update({
#         "cv_id": cv_id,
#         "session_id": session_id,
#         "jd_id": jd_id,
#         "parsed_text": parsed_text,
#         "status": "pending",
#         "created_at": datetime.utcnow(),
#     })

#     # Save CV
#     await cvs_coll.insert_one(cv_data)

#     # Link CV to JD
#     await jds_coll.update_one(
#         {"jd_id": jd_id},
#         {"$push": {"mapped_cvs": cv_id}}
#     )

#     # Track in session
#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {"$push": {"added_cvs": cv_id}}
#     )

#     return CVModel(**cv_data)

# @router.post("/middleware", response_model=CVModel)
# async def add_cv_from_middleware(session_id: str, jd_id: str, cv: MiddlewareCVInput):
#     """Add CV from middleware payload format"""
#     # Ensure JD exists
#     jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
#     if not jd:
#         raise HTTPException(status_code=404, detail="JD not found for this session")

#     # Extract text from base64_content
#     extracted_text = ""
#     try:
#         if cv.base64_content:
#             extracted_text = decode_base64_to_text(cv.base64_content)
#     except Exception as e:
#         # Log the error but continue - we can still store the CV
#         print(f"Warning: Could not extract text from CV {cv.cv_id}: {e}")

#     cv_data = {
#         "cv_id": cv.cv_id,  # Use the ID provided by middleware
#         "session_id": session_id,
#         "jd_id": jd_id,
#         "filename": cv.filename,
#         "original_base64": cv.base64_content,  # Map base64_content to original_base64
#         "extracted_text": extracted_text,
#         "Profile": None,
#         "processed": False,
#         "created_at": datetime.utcnow(),
#     }

#     # Save CV
#     await cvs_coll.insert_one(cv_data)

#     # Link CV to JD
#     await jds_coll.update_one(
#         {"jd_id": jd_id},
#         {"$push": {"mapped_cvs": cv.cv_id}}
#     )

#     # Track in session
#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {"$push": {"added_cvs": cv.cv_id}}
#     )

#     return CVModel(**cv_data)

# @router.get("/", response_model=list[CVModel])
# async def list_cvs(session_id: str, jd_id: str):
#     """Get all CVs for a JD in a session"""
#     cursor = cvs_coll.find({"session_id": session_id, "jd_id": jd_id})
#     cvs = await cursor.to_list(length=None)
#     return [CVModel(**cv) for cv in cvs]

# New comprehensive GET endpoints

@router.get("/all-cvs", response_model=List[CVModel])
async def list_all_cvs():
    """Get all CVs in the system"""
    cursor = cvs_coll.find({})
    cvs = await cursor.to_list(length=None)
    return [CVModel(**cv) for cv in cvs]

@router.get("/jd/{jd_id}", response_model=List[CVModel])
async def list_cvs_by_jd(jd_id: str):
    """Get all CVs for a specific JD across all sessions"""
    cursor = cvs_coll.find({"jd_id": jd_id})
    cvs = await cursor.to_list(length=None)
    return [CVModel(**cv) for cv in cvs]

@router.get("/session/{session_id}", response_model=List[CVModel])
async def list_cvs_by_session(session_id: str):
    """Get all CVs for a specific session across all JDs"""
    cursor = cvs_coll.find({"session_id": session_id})
    cvs = await cursor.to_list(length=None)
    return [CVModel(**cv) for cv in cvs]

@router.get("/status", response_model=List[CVModel])
async def list_cvs_by_status(status: str = Query(..., description="CV processing status: 'true or false'")):
    """Get all CVs with a specific processing status"""
    status_lower = status.lower()
    
    # Query based on the processed boolean field
    if status_lower == "true":
        query = {"processed": True}
    elif status_lower == "false":
        query = {"processed": False}
    else:
        # For any other status, return empty result or raise error
        raise HTTPException(status_code=400, detail="Status must be 'true' or 'false'")

    cursor = cvs_coll.find(query)
    cvs = await cursor.to_list(length=None)
    return [CVModel(**cv) for cv in cvs]