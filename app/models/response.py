# models/responses.py
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class MatchResult(BaseModel):
    jd_id: str
    cv_id: str
    category: str
    total_score: float
    sim_embed: float
    skill_coverage: float
    must_have_penalty: float
    llm_consistency: float
    rationale: str


class MatchReport(BaseModel):
    match_report_id: str
    session_id: str
    jd_id: str
    summary: str
    match_results: List[MatchResult]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IncrementalMatchResponse(BaseModel):
    session_id: str
    jd_id: str
    new_results: List[MatchResult]
    count: int