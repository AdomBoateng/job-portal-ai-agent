from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# -------- Sessions --------
class SessionModel(BaseModel):
    session_id: str
    status: str = "pending"   # pending, active, processing, completed, expired, failed
    jd_ids: List[str] = []
    added_cvs: List[str] = []
    processed_cvs: List[str] = []
    reports_generated: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

# -------- Job Descriptions --------
class JDModel(BaseModel):
    jd_id: str
    session_id: str
    title: str
    description: str
    skills: List[str] = []
    responsibilities: List[str] = []
    mapped_cvs: List[str] = []
    match_report_id: Optional[str] = None
    status: str = "active"  # active, completed, expired, failed
    completed_at: Optional[datetime] = None
    completion_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

# -------- CVs --------
class CVModel(BaseModel):
    cv_id: str
    session_id: str
    jd_id: str
    filename: str
    original_base64: str   # can be None after decoding
    extracted_text: str
    Profile: Optional[str] = None
    processed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

# -------- Reports --------
class ReportDetail(BaseModel):
    cv_id: str
    score: float
    reasoning: str

class ReportModel(BaseModel):
    match_report_id: str
    session_id: str
    jd_id: str
    summary: str
    details: List[ReportDetail] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)