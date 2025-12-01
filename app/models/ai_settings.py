"""
AI Settings Models for Configuration Management
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Available model types"""
    LLM = "llm"
    EMBEDDING = "embedding"

class AlgorithmType(str, Enum):
    """Available matching algorithms"""
    JACCARD = "jaccard"
    WEIGHTED_SKILL = "weighted_skill"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    HYBRID = "hybrid"

class LLMSettings(BaseModel):
    """LLM Configuration Settings"""
    model_name: str = Field(default="llava:7b", description="LLM model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

class EmbeddingSettings(BaseModel):
    """Embedding Model Configuration"""
    model_name: str = Field(default="nomic-embed-text:latest", description="Embedding model name")
    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    dimension: Optional[int] = Field(default=None, description="Embedding dimension")
    timeout: int = Field(default=30, ge=1, le=300, description="Request timeout in seconds")

class ScoringThresholds(BaseModel):
    """Matching Score Thresholds"""
    select_min: float = Field(default=0.72, ge=0.0, le=1.0, description="Minimum score for automatic selection")
    reject_max: float = Field(default=0.48, ge=0.0, le=1.0, description="Maximum score for automatic rejection")
    strong_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Strong match threshold")
    moderate_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Moderate match threshold")
    
    @validator('reject_max')
    def validate_thresholds(cls, v, values):
        if 'select_min' in values and v >= values['select_min']:
            raise ValueError('reject_max must be less than select_min')
        return v

class SkillWeights(BaseModel):
    """Custom skill importance weights"""
    default_weight: float = Field(default=1.0, ge=0.0, description="Default weight for unspecified skills")
    skill_weights: Dict[str, float] = Field(default_factory=dict, description="Custom weights for specific skills")
    must_have_skills: List[str] = Field(default_factory=list, description="Critical skills that heavily impact scoring")
    
    @validator('skill_weights')
    def validate_weights(cls, v):
        for skill, weight in v.items():
            if weight < 0:
                raise ValueError(f'Weight for skill "{skill}" must be non-negative')
        return v

class MatchingAlgorithm(BaseModel):
    """Matching Algorithm Configuration"""
    primary_algorithm: AlgorithmType = Field(default=AlgorithmType.HYBRID, description="Primary matching algorithm")
    use_embedding_similarity: bool = Field(default=True, description="Include embedding similarity in scoring")
    use_skill_coverage: bool = Field(default=True, description="Include skill coverage in scoring")
    use_llm_consistency: bool = Field(default=True, description="Include LLM consistency check")
    
    # Algorithm weights for hybrid approach
    embedding_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Weight for embedding similarity")
    skill_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for skill coverage")
    consistency_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for LLM consistency")
    
    @validator('consistency_weight')
    def validate_total_weights(cls, v, values):
        total = v + values.get('embedding_weight', 0) + values.get('skill_weight', 0)
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError('Algorithm weights must sum to 1.0')
        return v

class PromptSettings(BaseModel):
    """AI Prompt Configuration"""
    extract_prompt: str = Field(
        default="""You are an information extractor.
Given the document below (CV or JD), return strict JSON with keys:
name, email, phone, years_experience, roles, skills, tools, domains, certifications, summary.

- Normalize skills/tools/domains to short lowercase tokens.
- If unknown, use null or empty list.
- Keep summary to 3–5 concise sentences.

DOCUMENT:
{doc}
""",
        description="Prompt for extracting information from CVs/JDs"
    )
    
    consistency_prompt: str = Field(
        default="""You are a recruiter assistant. Score how well this CV matches the JD on a 0–1 scale.
Return JSON: {{"score": <0..1>, "why": "<1-2 sentences>"}}

JD SUMMARY:
{jd_summary}

JD KEY SKILLS:
{jd_skills}

CV SUMMARY:
{cv_summary}

CV KEY SKILLS:
{cv_skills}
""",
        description="Prompt for CV-JD consistency evaluation"
    )
    
    version: str = Field(default="1.0", description="Prompt version for tracking changes")

class ProcessingSettings(BaseModel):
    """Processing and Performance Configuration"""
    batch_size: int = Field(default=10, ge=1, le=100, description="Batch size for processing CVs")
    max_concurrent: int = Field(default=5, ge=1, le=20, description="Maximum concurrent AI requests")
    retry_attempts: int = Field(default=3, ge=0, le=10, description="Number of retry attempts for failed requests")
    retry_delay: float = Field(default=1.0, ge=0.1, le=60.0, description="Delay between retries in seconds")
    cache_results: bool = Field(default=True, description="Cache AI processing results")
    cache_ttl: int = Field(default=3600, ge=60, description="Cache TTL in seconds")

class AISettings(BaseModel):
    """Complete AI Agent Settings Configuration"""
    setting_id: str = Field(description="Unique identifier for this settings configuration")
    name: str = Field(description="Human-readable name for this configuration")
    description: Optional[str] = Field(default=None, description="Description of this configuration")
    
    # Core AI Configuration
    llm_settings: LLMSettings = Field(default_factory=LLMSettings)
    embedding_settings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    scoring_thresholds: ScoringThresholds = Field(default_factory=ScoringThresholds)
    skill_weights: SkillWeights = Field(default_factory=SkillWeights)
    matching_algorithm: MatchingAlgorithm = Field(default_factory=MatchingAlgorithm)
    prompt_settings: PromptSettings = Field(default_factory=PromptSettings)
    processing_settings: ProcessingSettings = Field(default_factory=ProcessingSettings)
    
    # Metadata
    is_active: bool = Field(default=False, description="Whether this configuration is currently active")
    is_default: bool = Field(default=False, description="Whether this is the default configuration")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None, description="User who created this configuration")
    tags: List[str] = Field(default_factory=list, description="Tags for categorizing configurations")

class AISettingsUpdate(BaseModel):
    """Model for updating AI settings (excludes read-only fields)"""
    name: Optional[str] = None
    description: Optional[str] = None
    llm_settings: Optional[LLMSettings] = None
    embedding_settings: Optional[EmbeddingSettings] = None
    scoring_thresholds: Optional[ScoringThresholds] = None
    skill_weights: Optional[SkillWeights] = None
    matching_algorithm: Optional[MatchingAlgorithm] = None
    prompt_settings: Optional[PromptSettings] = None
    processing_settings: Optional[ProcessingSettings] = None
    is_active: Optional[bool] = None
    tags: Optional[List[str]] = None

class ModelStatus(BaseModel):
    """Model availability status"""
    model_name: str
    model_type: ModelType
    is_available: bool
    status: str
    last_checked: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

class SettingsValidationResult(BaseModel):
    """Result of settings validation"""
    is_valid: bool
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    test_results: Dict[str, Any] = Field(default_factory=dict)
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)