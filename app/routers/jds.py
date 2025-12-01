from fastapi import APIRouter, Query
from app.services.db import jds_coll  #sessions_coll
from app.models.schemas import JDModel
# from app.models.middleware_schemas import MiddlewareJDInput
# from datetime import datetime
from typing import List
# import uuid

router = APIRouter()

# @router.post("/", response_model=JDModel)
# async def add_jd(session_id: str, jd: JDModel):
#     """Add JD to a session"""
#     jd_id = str(uuid.uuid4())
#     jd_data = jd.dict()
#     jd_data.update({
#         "jd_id": jd_id,
#         "session_id": session_id,
#         "created_at": datetime.utcnow(),
#     })

#     # Save JD
#     await jds_coll.insert_one(jd_data)

#     # Link JD to session
#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {"$push": {"jd_ids": jd_id}}
#     )

#     return JDModel(**jd_data)

# @router.post("/middleware", response_model=JDModel)
# async def add_jd_from_middleware(session_id: str, jd: MiddlewareJDInput):
#     """Add JD from middleware payload format"""
#     jd_data = {
#         "jd_id": jd.jd_id,  # Use the ID provided by middleware
#         "session_id": session_id,
#         "title": jd.title,
#         "description": jd.description,
#         "skills": jd.skills,
#         "responsibilities": jd.responsibilities,
#         "mapped_cvs": [],
#         "match_report_id": None,
#         "created_at": datetime.utcnow(),
#     }

#     # Save JD
#     await jds_coll.insert_one(jd_data)

#     # Link JD to session
#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {"$push": {"jd_ids": jd.jd_id}}
#     )

#     return JDModel(**jd_data)

# @router.get("/", response_model=list[JDModel])
# async def list_jds(session_id: str):
#     """Get all JDs for a session"""
#     cursor = jds_coll.find({"session_id": session_id})
#     jds = await cursor.to_list(length=None)
#     return [JDModel(**jd) for jd in jds]

# New comprehensive GET endpoints

@router.get("/all", response_model=List[JDModel])
async def list_all_jds():
    """Get all JDs in the system"""
    cursor = jds_coll.find({})
    jds = await cursor.to_list(length=None)
    return [JDModel(**jd) for jd in jds]

@router.get("/{jd_id}", response_model=List[JDModel])
async def list_jds_by_jd_id(jd_id: str):
    """Get all JDs for a specific JD ID"""
    cursor = jds_coll.find({"jd_id": jd_id})
    jds = await cursor.to_list(length=None)
    return [JDModel(**jd) for jd in jds]

@router.get("/session/{session_id}", response_model=List[JDModel])
async def list_jds_by_session(session_id: str):
    """Get all JDs for a specific session"""
    cursor = jds_coll.find({"session_id": session_id})
    jds = await cursor.to_list(length=None)
    return [JDModel(**jd) for jd in jds]

@router.get("/status", response_model=List[JDModel])
async def list_jds_by_status(status: str = Query(..., description="JD status (e.g., 'active', 'inactive', 'completed')")):
    """Get all JDs with a specific status"""
    cursor = jds_coll.find({"status": status})
    jds = await cursor.to_list(length=None)
    return [JDModel(**jd) for jd in jds]