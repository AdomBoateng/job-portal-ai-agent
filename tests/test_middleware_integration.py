import pytest
import base64
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from datetime import datetime

@pytest.fixture
def test_app():
    from app.main import app
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

class TestMiddlewareIntegration:
    """Test the complete middleware integration flow"""
    
    def test_middleware_payload_structure(self):
        """Test that middleware payload structure is valid"""
        from app.models.middleware_schemas import InitialSessionPayload, MiddlewareJDInput, MiddlewareCVInput
        
        # Test middleware payload structure
        jd_data = {
            "jd_id": "jd-123",
            "title": "Senior Python Developer",
            "description": "We are seeking an experienced Python developer...",
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "responsibilities": ["Develop backend APIs", "Design database schemas"]
        }
        
        cv_data = {
            "cv_id": "cv-001",
            "jd_id": "jd-123",
            "filename": "John_Doe.pdf",
            "base64_content": base64.b64encode(b"Sample CV content").decode()
        }
        
        payload_data = {
            "session_id": "session-abc-123",
            "job_descriptions": [jd_data],
            "cvs": [cv_data]
        }
        
        # These should not raise validation errors
        jd_input = MiddlewareJDInput(**jd_data)
        cv_input = MiddlewareCVInput(**cv_data)
        payload = InitialSessionPayload(**payload_data)
        
        assert jd_input.jd_id == "jd-123"
        assert cv_input.cv_id == "cv-001"
        assert payload.session_id == "session-abc-123"
        assert len(payload.job_descriptions) == 1
        assert len(payload.cvs) == 1

    @patch('app.routers.middleware.sessions_coll')
    @patch('app.routers.middleware.jds_coll')
    @patch('app.routers.middleware.cvs_coll')
    def test_complete_flow_phase_1_session_creation(self, mock_cvs_coll, mock_jds_coll, mock_sessions_coll, client):
        """Test Phase 1: Complete session creation from middleware payload"""
        
        # Mock database operations
        mock_sessions_coll.find_one = AsyncMock(return_value=None)  # Session doesn't exist
        mock_sessions_coll.insert_one = AsyncMock()
        mock_sessions_coll.update_one = AsyncMock()
        mock_jds_coll.insert_one = AsyncMock()
        mock_jds_coll.update_one = AsyncMock()
        mock_cvs_coll.insert_one = AsyncMock()
        mock_sessions_coll.find_one.side_effect = [
            None,  # First call: session doesn't exist
            {      # Second call: return created session
                "session_id": "session-abc-123",
                "status": "pending",
                "jd_ids": ["jd-123"],
                "added_cvs": ["cv-001", "cv-002"],
                "processed_cvs": [],
                "reports_generated": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        # Middleware payload (exact format from documentation)
        payload = {
            "session_id": "session-abc-123",
            "job_descriptions": [
                {
                    "jd_id": "jd-123",
                    "title": "Senior Python Developer",
                    "description": "We are seeking an experienced Python developer...",
                    "skills": ["Python", "Django", "PostgreSQL", "AWS"],
                    "responsibilities": ["Develop backend APIs", "Design database schemas"]
                }
            ],
            "cvs": [
                {
                    "cv_id": "cv-001",
                    "jd_id": "jd-123",
                    "filename": "John_Doe.pdf",
                    "base64_content": base64.b64encode(b"John Doe CV content").decode()
                },
                {
                    "cv_id": "cv-002",
                    "jd_id": "jd-123", 
                    "filename": "Jane_Smith.pdf",
                    "base64_content": base64.b64encode(b"Jane Smith CV content").decode()
                }
            ]
        }
        
        response = client.post("/middleware/sessions", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session-abc-123"
        assert data["status"] == "pending"
        assert "jd-123" in data["jd_ids"]
        assert "cv-001" in data["added_cvs"]
        assert "cv-002" in data["added_cvs"]
        assert not data["reports_generated"]

    @patch('app.routers.reports.jds_coll')
    @patch('app.routers.reports.cvs_coll')
    @patch('app.routers.reports.reports_coll')
    @patch('app.routers.reports.sessions_coll')
    @patch('app.routers.reports._do_matching_sync')
    def test_complete_flow_phase_2_ai_processing(self, mock_matching, mock_sessions_coll, mock_reports_coll, mock_cvs_coll, mock_jds_coll, client):
        """Test Phase 2: AI processing and matching"""
        
        # Mock JD document
        jd_doc = {
            "jd_id": "jd-123",
            "session_id": "session-abc-123",
            "title": "Senior Python Developer",
            "description": "We are seeking an experienced Python developer...",
            "skills": ["Python", "Django", "PostgreSQL", "AWS"],
            "responsibilities": ["Develop backend APIs", "Design database schemas"]
        }
        
        # Mock CV documents
        cv_docs = [
            {
                "cv_id": "cv-001",
                "session_id": "session-abc-123",
                "jd_id": "jd-123",
                "filename": "John_Doe.pdf",
                "extracted_text": "John Doe Senior Software Engineer 5 years Python experience..."
            },
            {
                "cv_id": "cv-002", 
                "session_id": "session-abc-123",
                "jd_id": "jd-123",
                "filename": "Jane_Smith.pdf",
                "extracted_text": "Jane Smith Full Stack Developer 3 years experience..."
            }
        ]
        
        # Mock AI matching results
        mock_report_doc = {
            "match_report_id": "rep_jd-123_1696204800",
            "session_id": "session-abc-123",
            "jd_id": "jd-123",
            "summary": "AI match for JD jd-123: 2 CVs evaluated.",
            "match_results": [
                {
                    "jd_id": "jd-123",
                    "cv_id": "cv-001",
                    "category": "strong",
                    "total_score": 0.87,
                    "sim_embed": 0.92,
                    "skill_coverage": 0.88,
                    "must_have_penalty": 0.0,
                    "llm_consistency": 0.85,
                    "rationale": "Excellent Python and Django experience..."
                },
                {
                    "jd_id": "jd-123",
                    "cv_id": "cv-002",
                    "category": "moderate",
                    "total_score": 0.68,
                    "sim_embed": 0.78,
                    "skill_coverage": 0.65,
                    "must_have_penalty": 0.1,
                    "llm_consistency": 0.72,
                    "rationale": "Good Python skills but lacks Django experience"
                }
            ],
            "created_at": datetime.utcnow()
        }
        
        # Setup mocks
        mock_jds_coll.find_one = AsyncMock(return_value=jd_doc)
        mock_cvs_coll.find.return_value.to_list = AsyncMock(return_value=cv_docs)
        mock_matching.return_value = mock_report_doc
        mock_reports_coll.insert_one = AsyncMock()
        mock_jds_coll.update_one = AsyncMock()
        mock_sessions_coll.update_one = AsyncMock()
        mock_cvs_coll.update_many = AsyncMock()
        
        # Trigger AI processing
        response = client.post("/sessions/session-abc-123/jds/jd-123/match")
        
        assert response.status_code == 200
        data = response.json()
        assert data["match_report_id"] == "rep_jd-123_1696204800"
        assert data["session_id"] == "session-abc-123"
        assert data["jd_id"] == "jd-123"
        assert len(data["match_results"]) == 2
        
        # Verify strong match
        strong_match = next(r for r in data["match_results"] if r["cv_id"] == "cv-001")
        assert strong_match["category"] == "strong"
        assert strong_match["total_score"] == 0.87
        
        # Verify moderate match
        moderate_match = next(r for r in data["match_results"] if r["cv_id"] == "cv-002")
        assert moderate_match["category"] == "moderate"
        assert moderate_match["total_score"] == 0.68

    @patch('app.routers.reports.sessions_coll')
    @patch('app.routers.reports.jds_coll')
    @patch('app.routers.reports.reports_coll')
    def test_complete_flow_phase_3_report_retrieval(self, mock_reports_coll, mock_jds_coll, mock_sessions_coll, client):
        """Test Phase 3: Final report retrieval by middleware"""
        
        # Mock session exists
        mock_sessions_coll.find_one = AsyncMock(return_value={"session_id": "session-abc-123"})
        
        # Mock JD exists
        mock_jds_coll.find_one = AsyncMock(return_value={"jd_id": "jd-123", "session_id": "session-abc-123"})
        
        # Mock report exists
        mock_report = {
            "match_report_id": "rep_jd-123_1696204800",
            "session_id": "session-abc-123",
            "jd_id": "jd-123",
            "summary": "AI match for JD jd-123: 2 CVs evaluated.",
            "match_results": [
                {
                    "jd_id": "jd-123",
                    "cv_id": "cv-001",
                    "category": "strong",
                    "total_score": 0.87,
                    "sim_embed": 0.92,
                    "skill_coverage": 0.88,
                    "must_have_penalty": 0.0,
                    "llm_consistency": 0.85,
                    "rationale": "Excellent Python and Django experience..."
                }
            ],
            "created_at": datetime.utcnow()
        }
        mock_reports_coll.find_one = AsyncMock(return_value=mock_report)
        
        # Get final report
        response = client.get("/sessions/session-abc-123/jds/jd-123/reports")
        
        assert response.status_code == 200
        data = response.json()
        assert data["match_report_id"] == "rep_jd-123_1696204800"
        assert data["session_id"] == "session-abc-123"
        assert data["jd_id"] == "jd-123"
        assert len(data["match_results"]) == 1
        assert data["match_results"][0]["category"] == "strong"

    @patch('app.routers.middleware.sessions_coll')
    @patch('app.routers.middleware.jds_coll')
    @patch('app.routers.middleware.cvs_coll')
    def test_incremental_cv_addition(self, mock_cvs_coll, mock_jds_coll, mock_sessions_coll, client):
        """Test adding CVs to existing session (incremental flow)"""
        
        # Mock session exists
        mock_sessions_coll.find_one = AsyncMock(return_value={"session_id": "session-abc-123"})
        
        # Mock JD exists
        mock_jds_coll.find_one = AsyncMock(return_value={"jd_id": "jd-123", "session_id": "session-abc-123"})
        
        # Mock database operations
        mock_cvs_coll.insert_one = AsyncMock()
        mock_jds_coll.update_one = AsyncMock()
        mock_sessions_coll.update_one = AsyncMock()
        
        # Incremental CV payload
        payload = {
            "cvs": [
                {
                    "cv_id": "cv-003",
                    "jd_id": "jd-123",
                    "filename": "Bob_Wilson.pdf",
                    "base64_content": base64.b64encode(b"Bob Wilson CV content").decode()
                }
            ]
        }
        
        response = client.post("/middleware/sessions/session-abc-123/add-cvs", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "Added 1 CVs to session session-abc-123" in data["message"]
        assert "cv-003" in data["cv_ids"]

    def test_base64_text_extraction(self):
        """Test automatic base64 text extraction functionality"""
        from app.routers.middleware import decode_base64_to_text
        
        # Test successful decoding
        original_text = "This is a sample CV content"
        base64_content = base64.b64encode(original_text.encode()).decode()
        extracted_text = decode_base64_to_text(base64_content)
        
        assert extracted_text == original_text
        
        # Test malformed base64 (should handle gracefully)
        invalid_base64 = "clearly_not_base64_content_with_invalid_chars!"
        extracted_text = decode_base64_to_text(invalid_base64)
        
        # Should either return empty string or handle gracefully
        assert isinstance(extracted_text, str)  # Should at least return a string

    def test_external_id_preservation(self):
        """Test that external IDs from middleware are preserved"""
        from app.models.middleware_schemas import MiddlewareJDInput, MiddlewareCVInput
        
        # Test JD ID preservation
        jd_data = {
            "jd_id": "external-jd-12345",  # External ID
            "title": "Test Job",
            "description": "Test Description",
            "skills": ["Python"],
            "responsibilities": ["Code"]
        }
        
        jd_input = MiddlewareJDInput(**jd_data)
        assert jd_input.jd_id == "external-jd-12345"  # Should preserve external ID
        
        # Test CV ID preservation
        cv_data = {
            "cv_id": "external-cv-67890",  # External ID
            "jd_id": "external-jd-12345",
            "filename": "test.pdf",
            "base64_content": base64.b64encode(b"content").decode()
        }
        
        cv_input = MiddlewareCVInput(**cv_data)
        assert cv_input.cv_id == "external-cv-67890"  # Should preserve external ID