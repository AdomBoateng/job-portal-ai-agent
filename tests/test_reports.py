import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import uuid

@pytest.fixture
def test_app():
    from fastapi import FastAPI
    from app.routers import reports
    
    app = FastAPI()
    app.include_router(reports.router)
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

class TestReportsRouter:
    """Test cases for reports router"""
    
    @patch('app.routers.reports.jds_coll')
    @patch('app.routers.reports.cvs_coll')
    @patch('app.routers.reports.sessions_coll')
    def test_get_session_not_found(self, mock_sessions_coll, mock_cvs_coll, mock_jds_coll, client):
        """Test getting reports for non-existent session"""
        session_id = str(uuid.uuid4())
        
        # Mock session not found
        mock_sessions_coll.find_one = AsyncMock(return_value=None)
        
        response = client.get(f"/sessions/{session_id}/reports")
        
        assert response.status_code == 404
        
    @patch('app.routers.reports.jds_coll')
    @patch('app.routers.reports.cvs_coll')
    @patch('app.routers.reports.sessions_coll')
    def test_match_session_not_found(self, mock_sessions_coll, mock_cvs_coll, mock_jds_coll, client):
        """Test matching for non-existent session"""
        session_id = str(uuid.uuid4())
        
        # Mock session not found
        mock_sessions_coll.find_one = AsyncMock(return_value=None)
        
        response = client.post(f"/sessions/{session_id}/match")
        
        assert response.status_code == 404
        
    @patch('app.routers.reports.sessions_coll')
    def test_get_reports_empty_session(self, mock_sessions_coll, client):
        """Test getting reports for session with no JDs or CVs"""
        session_id = str(uuid.uuid4())
        
        # Mock session exists but empty
        mock_sessions_coll.find_one = AsyncMock(return_value={
            "session_id": session_id,
            "jd_ids": [],
            "cv_ids": []
        })
        
        response = client.get(f"/sessions/{session_id}/reports")
        
        # Should handle empty session gracefully
        assert response.status_code in [200, 404]  # Depending on implementation