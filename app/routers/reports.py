# routers/reports.py
# import asyncio
# import base64
import logging
# import os
# import tempfile
# from datetime import datetime
from typing import List #, Dict, Any

from fastapi import APIRouter, HTTPException

# DB collections
# from app.models.schemas import CVModel
from app.services.db import jds_coll, reports_coll, sessions_coll #,cvs_coll

# Import AI agent modules
# from app.services.graph import extract_profile, write_reports, llm_consistency, SELECT_MIN, REJECT_MAX
# from app.services.matching import rank_and_score, categorize
# from app.models.models import Profile
# import app.helpers.parsing as parsing

# Response models
from app.models.response import MatchReport, MatchResult # IncrementalMatchResponse
# from app.models.middleware_schemas import IncrementalCVPayload

router = APIRouter(prefix="/sessions", tags=["reports"])
logger = logging.getLogger(__name__)


# # ----------------------
# # Helper functions
# # ----------------------

# def _build_jd_profile_from_doc(jd_doc: Dict[str, Any]) -> Profile:
#     """Convert JD Mongo doc → Profile for AI matching pipeline."""
#     title = jd_doc.get("title") or ""
#     description = jd_doc.get("description") or ""
#     skills = [s.lower() for s in jd_doc.get("skills", []) if isinstance(s, str)]
#     summary = f"{title}. {description}".strip()

#     return Profile(
#         name=None,
#         email=None,
#         phone=None,
#         years_experience=None,
#         roles=[],
#         skills=skills,
#         tools=[],
#         domains=[],
#         certifications=[],
#         summary=summary,
#     )


# def _write_temp_file_from_base64(b64: str, filename_hint: str = "cv") -> str:
#     """Decode base64 → temporary file path."""
#     b = base64.b64decode(b64)
#     ext = os.path.splitext(filename_hint)[1] or ".pdf"
#     tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
#     tf.write(b)
#     tf.flush()
#     tf.close()
#     return tf.name


# def _extract_text_from_filepath(path: str, filename: str = "") -> str:
#     """Extract text using parsing utils."""
#     ext = os.path.splitext(filename or path)[1].lower()

#     try:
#         if ext == ".pdf":
#             return parsing.read_pdf(parsing.Path(path))
#         if ext == ".docx":
#             return parsing.read_docx(parsing.Path(path))
#         if ext == ".txt":
#             return parsing.read_txt(parsing.Path(path))
#     except Exception:
#         pass

#     try:
#         with open(path, "rb") as fh:
#             return fh.read().decode("utf-8", errors="ignore")
#     except Exception:
#         return ""


# def _cv_doc_to_profile_sync(cv_doc: Dict[str, Any]) -> Profile:
#     """Convert CV Mongo doc → Profile for AI matching pipeline."""
#     cv_id = cv_doc.get("cv_id")
#     filename = cv_doc.get("filename", f"{cv_id}.pdf")
#     parsed_text = cv_doc.get("parsed_text")
#     structured_profile = cv_doc.get("profile")

#     # If profile already stored
#     if structured_profile:
#         try:
#             return Profile(**structured_profile)
#         except Exception:
#             return Profile(
#                 name=structured_profile.get("name"),
#                 email=structured_profile.get("email"),
#                 phone=structured_profile.get("phone"),
#                 years_experience=structured_profile.get("years_experience"),
#                 roles=structured_profile.get("roles", []),
#                 skills=structured_profile.get("skills", []),
#                 tools=structured_profile.get("tools", []),
#                 domains=structured_profile.get("domains", []),
#                 certifications=structured_profile.get("certifications", []),
#                 summary=structured_profile.get("summary", parsed_text or ""),
#             )

#     # If raw text stored
#     if parsed_text:
#         from app.models.models import DocRecord
#         doc = DocRecord(id=cv_id, path=filename, text=parsed_text)
#         return extract_profile(doc)

#     # If base64 provided
#     b64 = cv_doc.get("original_base64") or cv_doc.get("raw_base64")
#     if b64:
#         tmp_path = _write_temp_file_from_base64(b64, filename_hint=filename)
#         try:
#             text = _extract_text_from_filepath(tmp_path, filename)
#             from app.models.models import DocRecord
#             doc = DocRecord(id=cv_id, path=tmp_path, text=text)
#             return extract_profile(doc)
#         finally:
#             if os.path.exists(tmp_path):
#                 os.remove(tmp_path)

#     # Fallback
#     return Profile(summary="", skills=[], roles=[], tools=[], domains=[], certifications=[])


# def _do_matching_sync(jd_doc: Dict[str, Any], cv_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Run AI agent pipeline and return a full report doc."""
#     jdid = jd_doc.get("jd_id") or str(jd_doc.get("_id"))
#     jd_prof = _build_jd_profile_from_doc(jd_doc)

#     # Build CV profiles
#     cv_profiles: Dict[str, Profile] = {}
#     for cv in cv_docs:
#         cv_id = cv.get("cv_id")
#         try:
#             prof = _cv_doc_to_profile_sync(cv)
#         except Exception as e:
#             logger.exception("CV extraction failed for %s: %s", cv_id, e)
#             prof = Profile(summary=cv.get("parsed_text", "") or "", skills=cv.get("skills", []))
#         cv_profiles[cv_id] = prof

#     # Run scoring
#     scored = rank_and_score(
#         jdid=jdid,
#         jd=jd_prof,
#         cv_map=cv_profiles,
#         must_skills=jd_prof.skills or [],
#         weights={},
#         llm_consistency_cb=llm_consistency,
#     )

#     for s in scored:
#         s.category = categorize(s.total_score, SELECT_MIN, REJECT_MAX)

#     # Collect results
#     match_results = []
#     rows_for_write = {}
#     for s in scored:
#         rec = {
#             "jd_id": s.jd_id,
#             "cv_id": s.cv_id,
#             "category": s.category,
#             "total_score": float(s.total_score),
#             "sim_embed": float(s.sim_embed),
#             "skill_coverage": float(s.skill_coverage),
#             "must_have_penalty": float(s.must_have_penalty),
#             "llm_consistency": float(s.llm_consistency),
#             "rationale": s.rationale,
#         }
#         match_results.append(rec)
#         rows_for_write[s.cv_id] = s

#     # Write CSV/Markdown reports
#     csv_path, md_path = write_reports(jdid, rows_for_write, jd_prof)

#     # Build report doc
#     match_report_id = f"rep_{jdid}_{int(datetime.utcnow().timestamp())}"
#     summary = f"AI match for JD {jdid}: {len(match_results)} CVs evaluated."

#     return {
#         "match_report_id": match_report_id,
#         "session_id": jd_doc.get("session_id"),
#         "jd_id": jdid,
#         "summary": summary,
#         "match_results": match_results,
#         "csv_path": csv_path,
#         "md_path": md_path,
#         "created_at": datetime.utcnow(),
#     }


# ----------------------
# API Route
# ----------------------

# @router.post("/{session_id}/jds/{jd_id}/match", response_model=MatchReport)
# async def generate_match_report(session_id: str, jd_id: str):
#     """Generate AI match report for a JD in a session."""
#     jd_doc = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
#     if not jd_doc:
#         raise HTTPException(status_code=404, detail="JD not found for session")

#     cv_docs = await cvs_coll.find({"jd_id": jd_id, "session_id": session_id}).to_list(length=None)
#     if not cv_docs:
#         raise HTTPException(status_code=400, detail="No CVs found for this JD in session")

#     loop = asyncio.get_running_loop()
#     report_doc = await loop.run_in_executor(None, _do_matching_sync, jd_doc, cv_docs)

#     # Save to DB
#     await reports_coll.insert_one(report_doc)
#     await jds_coll.update_one(
#         {"jd_id": jd_id, "session_id": session_id},
#         {"$set": {"match_report_id": report_doc["match_report_id"]}},
#     )
#     cv_ids = [r["cv_id"] for r in report_doc["match_results"]]
#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {"$addToSet": {"processed_cvs": {"$each": cv_ids}},
#          "$set": {"reports_generated": True, "updated_at": datetime.utcnow()}},
#     )
#     await cvs_coll.update_many(
#         {"cv_id": {"$in": cv_ids}, "session_id": session_id},
#         {"$set": {"status": "processed"}},
#     )

#     return MatchReport(
#         match_report_id=report_doc["match_report_id"],
#         session_id=report_doc["session_id"],
#         jd_id=report_doc["jd_id"],
#         summary=report_doc["summary"],
#         match_results=[MatchResult(**r) for r in report_doc["match_results"]],
#         created_at=report_doc["created_at"],
#     )

# @router.post("/{session_id}/jds/{jd_id}/add-cvs", response_model=IncrementalMatchResponse)
# async def add_cvs(session_id: str, jd_id: str, payload: IncrementalCVPayload):
#     """Add CVs from middleware format and run matching"""
#     new_cvs = payload.cvs
#     if not new_cvs:
#         raise HTTPException(400, "No CVs provided")

#     jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
#     if not jd:
#         raise HTTPException(404, "JD not found")

#     # Convert middleware CV format to internal format and insert into DB
#     cv_docs = []
#     for cv_input in new_cvs:
#         # Extract text from base64_content
#         extracted_text = ""
#         try:
#             if cv_input.base64_content:
#                 extracted_text = base64.b64decode(cv_input.base64_content).decode("utf-8", errors="ignore")
#         except Exception as e:
#             logger.warning(f"Could not extract text from CV {cv_input.cv_id}: {e}")

#         cv_data = {
#             "cv_id": cv_input.cv_id,
#             "session_id": session_id,
#             "jd_id": jd_id,
#             "filename": cv_input.filename,
#             "original_base64": cv_input.base64_content,  # Map base64_content to original_base64
#             "extracted_text": extracted_text,
#             "Profile": None,
#             "processed": False,
#             "created_at": datetime.utcnow(),
#         }
#         cv_docs.append(cv_data)

#     # Insert all CVs
#     await cvs_coll.insert_many(cv_docs)

#     # Update JD with new mapped CVs
#     cv_ids = [cv["cv_id"] for cv in cv_docs]
#     await jds_coll.update_one(
#         {"jd_id": jd_id, "session_id": session_id},
#         {"$addToSet": {"mapped_cvs": {"$each": cv_ids}}}
#     )

#     # Run matching only for these new CVs
#     loop = asyncio.get_running_loop()
#     report_doc = await loop.run_in_executor(None, _do_matching_sync, jd, cv_docs)
#     results = report_doc["match_results"]

#     # Append results to existing report
#     await reports_coll.update_one(
#         {"jd_id": jd_id, "session_id": session_id},
#         {"$push": {"match_results": {"$each": results}}},
#         upsert=True
#     )

#     await sessions_coll.update_one(
#         {"session_id": session_id},
#         {
#             "$set": {"updated_at": datetime.utcnow()},
#             "$addToSet": {
#                 "processed_cvs": {"$each": cv_ids},
#                 "added_cvs": {"$each": cv_ids}
#             }
#         }
#     )

#     await cvs_coll.update_many(
#         {"cv_id": {"$in": cv_ids}, "session_id": session_id},
#         {"$set": {"processed": True}}
#     )

#     return IncrementalMatchResponse(
#         session_id=session_id,
#         jd_id=jd_id,
#         new_results=[MatchResult(**r) for r in results],
#         count=len(results),
#     )

@router.get("/all", response_model=List[MatchReport])
async def list_all_reports():
    """Get all match reports in the system"""
    cursor = reports_coll.find({})
    reports = await cursor.to_list(length=None)
    return [MatchReport(**report) for report in reports]

@router.get("/{session_id}/jds/{jd_id}/reports", response_model=MatchReport)
async def get_match_report(session_id: str, jd_id: str):
    """Get the final AI match report for a JD in a session."""
    # Verify session exists
    session = await sessions_coll.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Verify JD exists in this session
    jd_doc = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
    if not jd_doc:
        raise HTTPException(status_code=404, detail="JD not found for session")
    
    # Get the report from reports collection
    report_doc = await reports_coll.find_one({"jd_id": jd_id, "session_id": session_id})
    if not report_doc:
        raise HTTPException(status_code=404, detail="No report found for this JD. Run matching first.")
    
    return MatchReport(
        match_report_id=report_doc["match_report_id"],
        session_id=report_doc["session_id"],
        jd_id=report_doc["jd_id"],
        summary=report_doc["summary"],
        match_results=[MatchResult(**r) for r in report_doc["match_results"]],
        created_at=report_doc["created_at"],
    )

@router.get("/jd/{jd_id}", response_model=List[MatchReport])
async def list_reports_by_jd(jd_id: str):
    """Get all match reports for a specific JD across all sessions"""
    cursor = reports_coll.find({"jd_id": jd_id})
    reports = await cursor.to_list(length=None)
    return [MatchReport(**report) for report in reports]

@router.get("/session/{session_id}", response_model=List[MatchReport])
async def list_reports_by_session(session_id: str):
    """Get all match reports for a specific session across all JDs"""
    cursor = reports_coll.find({"session_id": session_id})
    reports = await cursor.to_list(length=None)
    return [MatchReport(**report) for report in reports]

@router.get("/reports/{match_report_id}", response_model=MatchReport)
async def get_report_by_id(match_report_id: str):
    """Get a specific match report by its ID"""
    report = await reports_coll.find_one({"match_report_id": match_report_id})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return MatchReport(**report)
