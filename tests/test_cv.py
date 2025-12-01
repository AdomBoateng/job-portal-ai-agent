import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
import base64

@pytest.fixture
def test_app():
    from fastapi import FastAPI
    from app.routers import cvs
    
    app = FastAPI()
    app.include_router(cvs.router, prefix="/cvs")
    return app

@pytest.fixture
def client(test_app):
    return TestClient(test_app)

class TestCVRouter:
    """Test cases for CV router"""
    
    def test_decode_base64_to_text_success(self):
        """Test successful base64 decoding"""
        from app.routers.cvs import decode_base64_to_text
        
        # Create a test string and encode it
        test_text = "This is a test CV content"
        encoded = base64.b64encode(test_text.encode()).decode()
        
        result = decode_base64_to_text(encoded)
        assert result == test_text
        
    def test_decode_base64_to_text_invalid(self):
        """Test base64 decoding with invalid input"""
        from app.routers.cvs import decode_base64_to_text
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            decode_base64_to_text("invalid_base64!")
            
        assert exc_info.value.status_code == 400
        assert "Invalid base64 CV" in str(exc_info.value.detail)
    
    @patch('app.routers.cv.jds_coll')
    @patch('app.routers.cv.cvs_coll')
    @patch('app.routers.cv.sessions_coll')
    def test_add_cv_jd_not_found(self, mock_sessions_coll, mock_cvs_coll, mock_jds_coll, client):
        """Test adding CV when JD doesn't exist"""
        session_id = str(uuid.uuid4())
        jd_id = str(uuid.uuid4())
        
        # Mock JD not found
        mock_jds_coll.find_one = AsyncMock(return_value=None)
        
        cv_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "original_base64": base64.b64encode(b"CV content").decode()
        }
        
        response = client.post(f"/cvs/?session_id={session_id}&jd_id={jd_id}", json=cv_data)
        
        assert response.status_code == 404
        assert "JD not found" in response.json()["detail"]
        
    @patch('app.routers.cv.jds_coll')
    @patch('app.routers.cv.cvs_coll')
    @patch('app.routers.cv.sessions_coll')
    def test_add_cv_success(self, mock_sessions_coll, mock_cvs_coll, mock_jds_coll, client):
        """Test successfully adding a CV"""
        session_id = str(uuid.uuid4())
        jd_id = str(uuid.uuid4())
        
        # Mock JD exists
        mock_jds_coll.find_one = AsyncMock(return_value={"jd_id": jd_id, "session_id": session_id})
        mock_cvs_coll.insert_one = AsyncMock()
        mock_sessions_coll.update_one = AsyncMock()
        
        cv_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "original_base64": base64.b64encode(b"CV content").decode()
        }
        
        response = client.post(f"/cvs/?session_id={session_id}&jd_id={jd_id}", json=cv_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "cv_id" in data
        assert data["name"] == "John Doe"
        assert data["email"] == "john@example.com"