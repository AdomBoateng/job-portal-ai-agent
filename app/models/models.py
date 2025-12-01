from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class DocRecord(BaseModel):
    id: str
    path: str
    text: str

class Profile(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    years_experience: Optional[float] = None
    roles: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    summary: str = ""

class MatchInput(BaseModel):
    jds: Dict[str, Profile]
    cvs: Dict[str, Profile]

class MatchScore(BaseModel):
    jd_id: str
    cv_id: str
    sim_embed: float
    skill_coverage: float
    must_have_penalty: float
    llm_consistency: float
    total_score: float
    category: str
    rationale: str
