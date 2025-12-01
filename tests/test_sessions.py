import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import uuid
from datetime import datetime

# We'll need to create a test app instance
@pytest.fixture
def test_app():
    from fastapi import FastAPI
    from app.routers import sessions
    
    app = FastAPI()
    app.include_router(sessions.router, prefix="/sessions")
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

class TestSessionsRouter:
    """Test cases for sessions router"""
    
    @patch('app.routers.sessions.sessions_coll')
    def test_create_session(self, mock_sessions_coll, client):
        """Test creating a new session"""
        # Mock the database insert
        mock_sessions_coll.insert_one = AsyncMock()
        
        response = client.post("/sessions/")
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "status" in data
        assert "created_at" in data
        
    @patch('app.routers.sessions.sessions_coll')
    def test_get_session_found(self, mock_sessions_coll, client):
        """Test getting an existing session"""
        session_id = str(uuid.uuid4())
        mock_session_data = {
            "session_id": session_id,
            "status": "active",
            "created_at": datetime.utcnow(),
            "jd_ids": [],
            "cv_ids": []
        }
        
        mock_sessions_coll.find_one = AsyncMock(return_value=mock_session_data)
        
        response = client.get(f"/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["status"] == "active"
        
    @patch('app.routers.sessions.sessions_coll')
    def test_get_session_not_found(self, mock_sessions_coll, client):
        """Test getting a non-existent session"""
        session_id = str(uuid.uuid4())
        
        mock_sessions_coll.find_one = AsyncMock(return_value=None)
        
        response = client.get(f"/sessions/{session_id}")
        
        assert response.status_code == 404
        assert "Session not found" in response.json()["detail"]
        
    @patch('app.routers.sessions.sessions_coll')
    def test_update_session_status(self, mock_sessions_coll, client):
        """Test updating session status"""
        session_id = str(uuid.uuid4())
        updated_session = {
            "session_id": session_id,
            "status": "completed",
            "updated_at": datetime.utcnow()
        }
        
        mock_sessions_coll.find_one_and_update = AsyncMock(return_value=updated_session)
        
        response = client.patch(f"/sessions/{session_id}?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"