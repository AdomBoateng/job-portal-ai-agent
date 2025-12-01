from fastapi import APIRouter, HTTPException
from app.services.db import sessions_coll, jds_coll, cvs_coll, reports_coll
from app.models.schemas import SessionModel
from app.models.middleware_schemas import (
    InitialSessionPayload, 
    IncrementalCVPayload, 
    JDCompletionPayload
)
from app.services.session_manager import SessionManager
from datetime import datetime
import base64
import asyncio
import logging

# Import matching functionality from reports
from app.services.graph import extract_profile, write_reports, llm_consistency, SELECT_MIN, REJECT_MAX
from app.services.matching import rank_and_score, categorize
from app.models.models import Profile
import app.helpers.parsing as parsing
import os
import tempfile
from typing import List, Dict, Any

router = APIRouter(prefix="/middleware", tags=["middleware"])
logger = logging.getLogger(__name__)

def decode_base64_to_text(b64_string: str) -> str:
    """Helper function to decode base64 content to text"""
    try:
        decoded_bytes = base64.b64decode(b64_string)
        return decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _build_jd_profile_from_doc(jd_doc: Dict[str, Any]) -> Profile:
    """Convert JD Mongo doc → Profile for AI matching pipeline."""
    title = jd_doc.get("title") or ""
    description = jd_doc.get("description") or ""
    skills = [s.lower() for s in jd_doc.get("skills", []) if isinstance(s, str)]
    summary = f"{title}. {description}".strip()

    return Profile(
        name=None,
        email=None,
        phone=None,
        years_experience=None,
        roles=[],
        skills=skills,
        tools=[],
        domains=[],
        certifications=[],
        summary=summary,
    )

def _write_temp_file_from_base64(b64: str, filename_hint: str = "cv") -> str:
    """Decode base64 → temporary file path."""
    b = base64.b64decode(b64)
    ext = os.path.splitext(filename_hint)[1] or ".pdf"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tf.write(b)
    tf.flush()
    tf.close()
    return tf.name

def _extract_text_from_filepath(path: str, filename: str = "") -> str:
    """Extract text using parsing utils."""
    ext = os.path.splitext(filename or path)[1].lower()

    try:
        if ext == ".pdf":
            return parsing.read_pdf(parsing.Path(path))
        if ext == ".docx":
            return parsing.read_docx(parsing.Path(path))
        if ext == ".txt":
            return parsing.read_txt(parsing.Path(path))
    except Exception:
        pass

    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _cv_doc_to_profile_sync(cv_doc: Dict[str, Any]) -> Profile:
    """Convert CV Mongo doc → Profile for AI matching pipeline."""
    cv_id = cv_doc.get("cv_id")
    filename = cv_doc.get("filename", f"{cv_id}.pdf")
    parsed_text = cv_doc.get("parsed_text")
    structured_profile = cv_doc.get("profile")

    # If profile already stored
    if structured_profile:
        try:
            return Profile(**structured_profile)
        except Exception:
            return Profile(
                name=structured_profile.get("name"),
                email=structured_profile.get("email"),
                phone=structured_profile.get("phone"),
                years_experience=structured_profile.get("years_experience"),
                roles=structured_profile.get("roles", []),
                skills=structured_profile.get("skills", []),
                tools=structured_profile.get("tools", []),
                domains=structured_profile.get("domains", []),
                certifications=structured_profile.get("certifications", []),
                summary=structured_profile.get("summary", parsed_text or ""),
            )

    # If raw text stored
    if parsed_text:
        from app.models.models import DocRecord
        doc = DocRecord(id=cv_id, path=filename, text=parsed_text)
        return extract_profile(doc)

    # If base64 provided
    b64 = cv_doc.get("original_base64") or cv_doc.get("raw_base64")
    if b64:
        tmp_path = _write_temp_file_from_base64(b64, filename_hint=filename)
        try:
            text = _extract_text_from_filepath(tmp_path, filename)
            from app.models.models import DocRecord
            doc = DocRecord(id=cv_id, path=tmp_path, text=text)
            return extract_profile(doc)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # Fallback
    return Profile(summary="", skills=[], roles=[], tools=[], domains=[], certifications=[])

def _do_matching_sync(jd_doc: Dict[str, Any], cv_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run AI agent pipeline and return a full report doc."""
    jdid = jd_doc.get("jd_id") or str(jd_doc.get("_id"))
    jd_prof = _build_jd_profile_from_doc(jd_doc)

    # Build CV profiles
    cv_profiles: Dict[str, Profile] = {}
    for cv in cv_docs:
        cv_id = cv.get("cv_id")
        try:
            prof = _cv_doc_to_profile_sync(cv)
        except Exception as e:
            logger.exception("CV extraction failed for %s: %s", cv_id, e)
            prof = Profile(summary=cv.get("parsed_text", "") or "", skills=cv.get("skills", []))
        cv_profiles[cv_id] = prof

    # Run scoring
    scored = rank_and_score(
        jdid=jdid,
        jd=jd_prof,
        cv_map=cv_profiles,
        must_skills=jd_prof.skills or [],
        weights={},
        llm_consistency_cb=llm_consistency,
    )

    for s in scored:
        s.category = categorize(s.total_score, SELECT_MIN, REJECT_MAX)

    # Collect results
    match_results = []
    rows_for_write = {}
    for s in scored:
        rec = {
            "jd_id": s.jd_id,
            "cv_id": s.cv_id,
            "category": s.category,
            "total_score": float(s.total_score),
            "sim_embed": float(s.sim_embed),
            "skill_coverage": float(s.skill_coverage),
            "must_have_penalty": float(s.must_have_penalty),
            "llm_consistency": float(s.llm_consistency),
            "rationale": s.rationale,
        }
        match_results.append(rec)
        rows_for_write[s.cv_id] = s

    # Write CSV/Markdown reports
    csv_path, md_path = write_reports(jdid, rows_for_write, jd_prof)

    # Build report doc
    match_report_id = f"rep_{jdid}_{int(datetime.utcnow().timestamp())}"
    summary = f"AI match for JD {jdid}: {len(match_results)} CVs evaluated."

    return {
        "match_report_id": match_report_id,
        "session_id": jd_doc.get("session_id"),
        "jd_id": jdid,
        "summary": summary,
        "match_results": match_results,
        "csv_path": csv_path,
        "md_path": md_path,
        "created_at": datetime.utcnow(),
    }

@router.post("/sessions", response_model=SessionModel)
async def initialize_session_from_middleware(payload: InitialSessionPayload):
    """Initialize a complete session with JDs and CVs from middleware payload"""
    
    # Check if session already exists
    existing_session = await sessions_coll.find_one({"session_id": payload.session_id})
    if existing_session:
        raise HTTPException(status_code=400, detail="Session already exists")
    
    # Create session with status from middleware
    session_data = {
        "session_id": payload.session_id,
        "status": payload.status,  # Use status from middleware payload
        "jd_ids": [],
        "added_cvs": [],
        "processed_cvs": [],
        "reports_generated": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    await sessions_coll.insert_one(session_data)
    
    # Process JDs
    jd_ids = []
    for jd_input in payload.job_descriptions:
        jd_data = {
            "jd_id": jd_input.jd_id,
            "session_id": payload.session_id,
            "title": jd_input.title,
            "description": jd_input.description,
            "skills": jd_input.skills,
            "responsibilities": jd_input.responsibilities,
            "mapped_cvs": [],
            "match_report_id": None,
            "created_at": datetime.utcnow(),
        }
        
        # Use upsert to handle potential duplicates within the same session
        await jds_coll.replace_one(
            {"jd_id": jd_input.jd_id, "session_id": payload.session_id},
            jd_data,
            upsert=True
        )
        jd_ids.append(jd_input.jd_id)
    
    # Process CVs
    cv_ids = []
    cv_jd_mapping = {}  # Track which CVs belong to which JDs
    
    for cv_input in payload.cvs:
        # Extract text from base64
        extracted_text = decode_base64_to_text(cv_input.base64_content)
        
        cv_data = {
            "cv_id": cv_input.cv_id,
            "session_id": payload.session_id,
            "jd_id": cv_input.jd_id,
            "filename": cv_input.filename,
            "original_base64": cv_input.base64_content,
            "extracted_text": extracted_text,
            "Profile": None,
            "processed": False,
            "created_at": datetime.utcnow(),
        }
        
        # Use upsert to handle potential duplicates within the same session
        await cvs_coll.replace_one(
            {"cv_id": cv_input.cv_id, "session_id": payload.session_id},
            cv_data,
            upsert=True
        )
        cv_ids.append(cv_input.cv_id)
        
        # Track CV-JD mapping
        if cv_input.jd_id not in cv_jd_mapping:
            cv_jd_mapping[cv_input.jd_id] = []
        cv_jd_mapping[cv_input.jd_id].append(cv_input.cv_id)
    
    # Update JDs with mapped CVs
    for jd_id, mapped_cv_ids in cv_jd_mapping.items():
        await jds_coll.update_one(
            {"jd_id": jd_id, "session_id": payload.session_id},
            {"$addToSet": {"mapped_cvs": {"$each": mapped_cv_ids}}}
        )
    
    # Update session with JD and CV IDs
    await sessions_coll.update_one(
        {"session_id": payload.session_id},
        {
            "$addToSet": {
                "jd_ids": {"$each": jd_ids},
                "added_cvs": {"$each": cv_ids}
            },
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    # Generate reports for each JD that has CVs
    reports_generated = []
    
    logger.debug(f"CV-JD mapping: {cv_jd_mapping}")
    
    if cv_jd_mapping:  # Only generate reports if there are CVs
        logger.info(f"Starting report generation for {len(cv_jd_mapping)} JDs")
        for jd_id, mapped_cv_ids in cv_jd_mapping.items():
            logger.debug(f"Processing JD {jd_id} with CVs {mapped_cv_ids}")
            try:
                # Get JD document
                jd_doc = await jds_coll.find_one({"jd_id": jd_id, "session_id": payload.session_id})
                logger.debug(f"JD document found for {jd_id}: {jd_doc is not None}")
                
                # Get CVs for this JD
                cv_docs = await cvs_coll.find({"cv_id": {"$in": mapped_cv_ids}, "session_id": payload.session_id}).to_list(length=None)
                logger.debug(f"Found {len(cv_docs)} CV documents for JD {jd_id}")
                
                if cv_docs:
                    logger.info(f"Running AI matching for JD {jd_id} with {len(cv_docs)} CVs")
                    # Run matching in executor for CPU-intensive work
                    loop = asyncio.get_running_loop()
                    report_doc = await loop.run_in_executor(None, _do_matching_sync, jd_doc, cv_docs)
                    logger.info(f"Report generated for JD {jd_id} with {len(report_doc['match_results'])} results")
                    
                    # Save report to database
                    await reports_coll.insert_one(report_doc)
                    
                    # Update JD with report ID
                    await jds_coll.update_one(
                        {"jd_id": jd_id, "session_id": payload.session_id},
                        {"$set": {"match_report_id": report_doc["match_report_id"]}}
                    )
                    
                    
                    # Mark CVs as processed
                    await cvs_coll.update_many(
                        {"cv_id": {"$in": mapped_cv_ids}, "session_id": payload.session_id},
                        {"$set": {"processed": True}}
                    )
                    
                    reports_generated.append(f"Generated report for JD {jd_id} with {len(report_doc['match_results'])} matches")
                else:
                    logger.warning(f"No CV documents found for JD {jd_id}")

            except Exception as e:
                logger.error(f"Error generating report for JD {jd_id}: {str(e)}", exc_info=True)
                reports_generated.append(f"Failed to generate report for JD {jd_id}: {str(e)}")

        logger.info(f"Report generation completed. Generated {len(reports_generated)} reports")
        # Update session to mark reports as generated and CVs as processed
        await sessions_coll.update_one(
            {"session_id": payload.session_id},
            {
                "$addToSet": {"processed_cvs": {"$each": cv_ids}},
                "$set": {
                    "reports_generated": True, 
                    "status": payload.status,  # Keep middleware status
                    "updated_at": datetime.utcnow()
                }
            }
        )
        logger.debug("Session updated with reports_generated=True")
    else:
        logger.info("No CVs to process, skipping report generation")
    
    # Return updated session with report generation info - query AFTER all updates
    updated_session = await sessions_coll.find_one({"session_id": payload.session_id})
    logger.debug(f"Final session reports_generated status: {updated_session.get('reports_generated', False)}")
    session_model = SessionModel(**updated_session)
    
    # Add report generation info to response (if needed for debugging)
    if reports_generated:
        logger.info(f"Session {payload.session_id} created with reports: {reports_generated}")
    
    return session_model

@router.post("/sessions/{session_id}/jds/{jd_id}/add-cvs")
async def add_cvs_to_existing_session(session_id: str, jd_id: str, payload: IncrementalCVPayload):
    """Add CVs to a specific JD in an existing session from middleware payload and automatically generate reports"""
    
    # Verify session exists
    session = await sessions_coll.find_one({"session_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify JD exists in this session
    jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
    if not jd:
        raise HTTPException(
            status_code=404, 
            detail=f"JD {jd_id} not found in session {session_id}"
        )

    # Update session status based on middleware input
    await sessions_coll.update_one(
        {"session_id": session_id},
        {"$set": {"status": payload.status, "updated_at": datetime.utcnow()}}
    )

    cv_ids = []
    
    for cv_input in payload.cvs:
        # All CVs will be associated with the JD specified in the path parameter
        
        # Extract text from base64
        extracted_text = decode_base64_to_text(cv_input.base64_content)
        
        cv_data = {
            "cv_id": cv_input.cv_id,
            "session_id": session_id,
            "jd_id": jd_id,  # Use the JD ID from path parameter
            "filename": cv_input.filename,
            "original_base64": cv_input.base64_content,
            "extracted_text": extracted_text,
            "Profile": None,
            "processed": False,
            "created_at": datetime.utcnow(),
        }
        
        # Use upsert to handle potential duplicates within the same session
        await cvs_coll.replace_one(
            {"cv_id": cv_input.cv_id, "session_id": session_id},
            cv_data,
            upsert=True
        )
        cv_ids.append(cv_input.cv_id)

    # Update the specific JD with new mapped CVs
    await jds_coll.update_one(
        {"jd_id": jd_id, "session_id": session_id},
        {"$addToSet": {"mapped_cvs": {"$each": cv_ids}}}
    )

    # Update session with new CV IDs
    await sessions_coll.update_one(
        {"session_id": session_id},
        {
            "$addToSet": {"added_cvs": {"$each": cv_ids}},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )

    # Generate report for the specific JD
    reports_generated = []
    
    try:
        # Get JD document
        jd_doc = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
        
        # Get only the new CVs for this JD
        new_cv_docs = await cvs_coll.find({"cv_id": {"$in": cv_ids}, "session_id": session_id}).to_list(length=None)
        
        if new_cv_docs:
            # Run matching in executor for CPU-intensive work
            loop = asyncio.get_running_loop()
            report_doc = await loop.run_in_executor(None, _do_matching_sync, jd_doc, new_cv_docs)
            results = report_doc["match_results"]

            # Check if report already exists for this JD
            existing_report = await reports_coll.find_one({"jd_id": jd_id, "session_id": session_id})
            
            if existing_report:
                # Append results to existing report
                await reports_coll.update_one(
                    {"jd_id": jd_id, "session_id": session_id},
                    {"$push": {"match_results": {"$each": results}}}
                )
                reports_generated.append(f"Appended {len(results)} results to existing report for JD {jd_id}")
            else:
                # Create new report
                await reports_coll.insert_one(report_doc)
                await jds_coll.update_one(
                    {"jd_id": jd_id, "session_id": session_id},
                    {"$set": {"match_report_id": report_doc["match_report_id"]}}
                )
                reports_generated.append(f"Created new report for JD {jd_id} with {len(results)} results")

            # Mark CVs as processed
            await cvs_coll.update_many(
                {"cv_id": {"$in": cv_ids}, "session_id": session_id},
                {"$set": {"processed": True}}
            )

    except Exception as e:
        logger.error(f"Failed to generate report for JD {jd_id}: {str(e)}")
        reports_generated.append(f"Failed to generate report for JD {jd_id}: {str(e)}")

    # Update session to mark reports as generated and CVs as processed
    await sessions_coll.update_one(
        {"session_id": session_id},
        {
            "$addToSet": {"processed_cvs": {"$each": cv_ids}},
            "$set": {
                "reports_generated": True, 
                "status": payload.status,  # Keep middleware status
                "updated_at": datetime.utcnow()
            }
        }
    )

    return {
        "message": f"Added {len(cv_ids)} CVs to JD {jd_id} in session {session_id} and processed reports",
        "cv_ids": cv_ids,
        "jd_id": jd_id,
        "reports_generated": reports_generated
    }

@router.post("/sessions/{session_id}/jds/{jd_id}/complete")
async def complete_jd_in_session(session_id: str, jd_id: str, payload: JDCompletionPayload):
    """
    Complete/expire a specific JD within a session.
    This endpoint is called by middleware to inform AI agent that a JD is completed/expired.
    """
    try:
        logger.info(f"Completing JD {jd_id} in session {session_id} with status {payload.status}")
        
        # Validate that the payload session_id matches the URL parameter
        if payload.session_id != session_id:
            raise HTTPException(
                status_code=400, 
                detail=f"Session ID mismatch: URL has {session_id}, payload has {payload.session_id}"
            )
        
        # Validate that the payload jd_id matches the URL parameter
        if payload.jd_id != jd_id:
            raise HTTPException(
                status_code=400, 
                detail=f"JD ID mismatch: URL has {jd_id}, payload has {payload.jd_id}"
            )
        
        # Complete the JD
        result = await SessionManager.complete_jd_in_session(
            session_id=session_id,
            jd_id=jd_id,
            status=payload.status,
            reason=payload.reason if payload.reason else None
        )
        
        logger.info(f"Successfully completed JD {jd_id} in session {session_id}")
        return result
        
    except ValueError as e:
        logger.warning(f"JD or session not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error completing JD {jd_id} in session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to complete JD: {str(e)}")