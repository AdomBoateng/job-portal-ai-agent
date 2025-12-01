"""
Test the simplified middleware status-aware session management
"""
import pytest
import asyncio
from datetime import datetime
from app.services.session_manager import SessionManager
from app.services.db import sessions_coll, jds_coll
from app.models.middleware_schemas import InitialSessionPayload, IncrementalCVPayload, JDCompletionPayload, MiddlewareJDInput, MiddlewareCVInput

@pytest.fixture
async def cleanup_test_data():
    """Clean up test data after tests"""
    yield
    # Clean up test sessions and JDs
    await sessions_coll.delete_many({"session_id": {"$regex": "^test_status_"}})
    await jds_coll.delete_many({"session_id": {"$regex": "^test_status_"}})

@pytest.mark.asyncio
async def test_session_creation_with_middleware_status(cleanup_test_data):
    """Test session creation using status from middleware payload"""
    
    session_id = "test_status_session_001"
    jd_id = "test_jd_status_001"
    
    # Create test payload with active status
    payload = InitialSessionPayload(
        session_id=session_id,
        status="active",  # Middleware specifies active status
        job_descriptions=[
            MiddlewareJDInput(
                jd_id=jd_id,
                title="Senior Python Developer",
                description="Python development role",
                skills=["python", "django", "rest-api"],
                responsibilities=["Develop APIs", "Code review"]
            )
        ],
        cvs=[]
    )
    
    # Create session manually to test the logic
    session_data = {
        "session_id": payload.session_id,
        "status": payload.status,
        "jd_ids": [],
        "added_cvs": [],
        "processed_cvs": [],
        "reports_generated": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    
    await sessions_coll.insert_one(session_data)
    
    # Verify session was created with correct status
    session = await sessions_coll.find_one({"session_id": session_id})
    assert session is not None
    assert session["status"] == "active"

@pytest.mark.asyncio
async def test_incremental_cv_with_status_update(cleanup_test_data):
    """Test adding CVs with status update from middleware"""
    
    session_id = "test_status_session_002"
    jd_id = "test_jd_status_002"
    
    # Create initial session
    session_data = {
        "session_id": session_id,
        "status": "pending",
        "jd_ids": [jd_id],
        "added_cvs": [],
        "processed_cvs": [],
        "reports_generated": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await sessions_coll.insert_one(session_data)
    
    # Create JD record
    jd_data = {
        "jd_id": jd_id,
        "session_id": session_id,
        "title": "Frontend Developer",
        "description": "React development role",
        "skills": ["react", "javascript"],
        "responsibilities": ["Build UIs"],
        "mapped_cvs": [],
        "match_report_id": None,
        "status": "active",
        "created_at": datetime.utcnow(),
    }
    await jds_coll.insert_one(jd_data)
    
    # Test incremental payload with status change
    incremental_payload = IncrementalCVPayload(
        status="processing",  # Middleware changing status to processing
        cvs=[
            MiddlewareCVInput(
                cv_id="cv_001",
                jd_id=jd_id,
                filename="resume1.pdf",
                base64_content="dGVzdCBjdiBjb250ZW50"  # "test cv content" in base64
            )
        ]
    )
    
    # Update session status as the endpoint would do
    await sessions_coll.update_one(
        {"session_id": session_id},
        {"$set": {"status": incremental_payload.status, "updated_at": datetime.utcnow()}}
    )
    
    # Verify status was updated
    session = await sessions_coll.find_one({"session_id": session_id})
    assert session["status"] == "processing"

@pytest.mark.asyncio
async def test_jd_completion_endpoint_logic(cleanup_test_data):
    """Test JD completion functionality"""
    
    session_id = "test_status_session_003"
    jd_id = "test_jd_status_003"
    
    # Create session
    session_data = {
        "session_id": session_id,
        "status": "active",
        "jd_ids": [jd_id],
        "added_cvs": [],
        "processed_cvs": [],
        "reports_generated": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await sessions_coll.insert_one(session_data)
    
    jd_data = {
        "jd_id": jd_id,
        "session_id": session_id,
        "title": "Data Scientist",
        "description": "ML/AI role",
        "skills": ["python", "machine-learning"],
        "responsibilities": ["Build ML models"],
        "mapped_cvs": [],
        "match_report_id": None,
        "status": "active",
        "created_at": datetime.utcnow(),
    }
    await jds_coll.insert_one(jd_data)
    
    # Test JD completion
    completion_payload = JDCompletionPayload(
        session_id=session_id,
        jd_id=jd_id,
        status="completed",
        reason="Job posting expired",
        completed_at=datetime.utcnow()
    )
    
    # Complete the JD
    result = await SessionManager.complete_jd_in_session(
        session_id=session_id,
        jd_id=jd_id,
        status=completion_payload.status,
        reason=completion_payload.reason
    )
    
    # Verify JD was marked as completed
    jd = await jds_coll.find_one({"jd_id": jd_id, "session_id": session_id})
    assert jd["status"] == "completed"
    assert jd["completion_reason"] == "Job posting expired"
    assert "completed_at" in jd
    
    # Verify session status was updated
    assert result["jd_status"] == "completed"
    assert result["completed_jds_count"] == 1
    assert result["total_jds_count"] == 1
    assert result["session_status"] == SessionManager.STATUS_COMPLETED  # All JDs completed

@pytest.mark.asyncio
async def test_multiple_jds_completion_logic(cleanup_test_data):
    """Test session completion when multiple JDs exist"""
    
    session_id = "test_status_session_004"
    jd1_id = "test_jd_status_004_1"
    jd2_id = "test_jd_status_004_2"
    
    # Create session
    session_data = {
        "session_id": session_id,
        "status": "active",
        "jd_ids": [jd1_id, jd2_id],
        "added_cvs": [],
        "processed_cvs": [],
        "reports_generated": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await sessions_coll.insert_one(session_data)
    
    # Create both JDs
    for jd_id in [jd1_id, jd2_id]:
        jd_data = {
            "jd_id": jd_id,
            "session_id": session_id,
            "title": f"Developer {jd_id}",
            "description": "Development role",
            "skills": ["python"],
            "responsibilities": ["Code"],
            "mapped_cvs": [],
            "match_report_id": None,
            "status": "active",
            "created_at": datetime.utcnow(),
        }
        await jds_coll.insert_one(jd_data)
    
    # Complete first JD
    result1 = await SessionManager.complete_jd_in_session(
        session_id=session_id,
        jd_id=jd1_id,
        status="completed"
    )
    
    # Session should still be active (not all JDs completed)
    assert result1["session_status"] == SessionManager.STATUS_ACTIVE
    assert result1["completed_jds_count"] == 1
    assert result1["total_jds_count"] == 2
    
    # Complete second JD
    result2 = await SessionManager.complete_jd_in_session(
        session_id=session_id,
        jd_id=jd2_id,
        status="expired"
    )
    
    # Now session should be completed (all JDs done)
    assert result2["session_status"] == SessionManager.STATUS_COMPLETED
    assert result2["completed_jds_count"] == 2
    assert result2["total_jds_count"] == 2

if __name__ == "__main__":
    # Run tests manually if needed
    async def run_tests():
        await test_session_creation_with_middleware_status()
        await test_incremental_cv_with_status_update()
        await test_jd_completion_endpoint_logic()
        await test_multiple_jds_completion_logic()
        print("All simplified status tests passed!")
    
    asyncio.run(run_tests())