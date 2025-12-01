import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import uuid
from datetime import datetime

@pytest.fixture
def test_app():
    from fastapi import FastAPI
    from app.routers import jds
    
    app = FastAPI()
    app.include_router(jds.router, prefix="/jds")
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

class TestJDsRouter:
    """Test cases for job descriptions router"""
    
    @patch('app.routers.jds.jds_coll')
    @patch('app.routers.jds.sessions_coll')
    def test_add_jd_success(self, mock_sessions_coll, mock_jds_coll, client):
        """Test successfully adding a JD to a session"""
        session_id = str(uuid.uuid4())
        
        # Mock database operations
        mock_jds_coll.insert_one = AsyncMock()
        mock_sessions_coll.update_one = AsyncMock()
        
        jd_data = {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "description": "We are looking for a software engineer...",
            "requirements": ["Python", "FastAPI", "MongoDB"],
            "location": "Remote"
        }
        
        response = client.post(f"/jds/?session_id={session_id}", json=jd_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "jd_id" in data
        assert data["title"] == "Software Engineer"
        assert data["company"] == "Tech Corp"
        assert data["session_id"] == session_id
        assert "created_at" in data
        
        # Verify database calls were made
        mock_jds_coll.insert_one.assert_called_once()
        mock_sessions_coll.update_one.assert_called_once()
        
    def test_add_jd_invalid_data(self, client):
        """Test adding JD with invalid data"""
        session_id = str(uuid.uuid4())
        
        # Missing required fields
        invalid_jd_data = {
            "title": "Software Engineer"
            # Missing other required fields
        }
        
        response = client.post(f"/jds/?session_id={session_id}", json=invalid_jd_data)
        
        # Should return validation error
        assert response.status_code == 422