"""
AI Settings Router - Comprehensive API for AI Agent Configuration Management
"""
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from app.services.db import ai_settings_coll
from app.models.ai_settings import (
    AISettings, AISettingsUpdate, ModelStatus, SettingsValidationResult,
    LLMSettings, EmbeddingSettings, ScoringThresholds, SkillWeights,
    MatchingAlgorithm, PromptSettings, ProcessingSettings, ModelType
)
from app.utils.utils import ollama_generate, ollama_embed

router = APIRouter(prefix="/ai-settings", tags=["ai-settings"])
logger = logging.getLogger(__name__)

# ==================== CORE SETTINGS CRUD OPERATIONS ====================

@router.post("/", response_model=AISettings)
async def create_settings(settings: AISettings):
    """Create a new AI settings configuration"""
    # Generate unique ID if not provided
    if not settings.setting_id:
        settings.setting_id = str(uuid.uuid4())
    
    # Check if setting_id already exists
    existing = await ai_settings_coll.find_one({"setting_id": settings.setting_id})
    if existing:
        raise HTTPException(status_code=400, detail="Settings with this ID already exists")
    
    # Ensure only one default configuration
    if settings.is_default:
        await ai_settings_coll.update_many(
            {"is_default": True},
            {"$set": {"is_default": False, "updated_at": datetime.utcnow()}}
        )
    
    # Ensure only one active configuration
    if settings.is_active:
        await ai_settings_coll.update_many(
            {"is_active": True},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )
    
    settings.created_at = datetime.utcnow()
    settings.updated_at = datetime.utcnow()
    
    await ai_settings_coll.insert_one(settings.dict())
    logger.info(f"Created AI settings configuration: {settings.setting_id}")
    return settings

@router.get("/", response_model=List[AISettings])
async def list_settings(
    active_only: bool = Query(False, description="Return only active settings"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip")
):
    """List all AI settings configurations with optional filtering"""
    query = {}
    
    if active_only:
        query["is_active"] = True
    
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",")]
        query["tags"] = {"$in": tag_list}
    
    cursor = ai_settings_coll.find(query).skip(offset).limit(limit).sort("created_at", -1)
    settings_list = await cursor.to_list(length=None)
    
    return [AISettings(**settings) for settings in settings_list]

@router.get("/{setting_id}", response_model=AISettings)
async def get_settings(setting_id: str):
    """Get a specific AI settings configuration by ID"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings configuration not found")
    
    return AISettings(**settings)

@router.get("/active/current", response_model=AISettings)
async def get_active_settings():
    """Get the currently active AI settings configuration"""
    settings = await ai_settings_coll.find_one({"is_active": True})
    if not settings:
        # Return default configuration if no active one exists
        settings = await ai_settings_coll.find_one({"is_default": True})
        if not settings:
            raise HTTPException(status_code=404, detail="No active or default AI settings found")
    
    return AISettings(**settings)

@router.put("/{setting_id}", response_model=AISettings)
async def update_settings(setting_id: str, updates: AISettingsUpdate):
    """Update an existing AI settings configuration"""
    existing = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not existing:
        raise HTTPException(status_code=404, detail="AI settings configuration not found")
    
    update_data = updates.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    # Handle exclusive flags
    if updates.is_default:
        await ai_settings_coll.update_many(
            {"is_default": True, "setting_id": {"$ne": setting_id}},
            {"$set": {"is_default": False, "updated_at": datetime.utcnow()}}
        )
    
    if updates.is_active:
        await ai_settings_coll.update_many(
            {"is_active": True, "setting_id": {"$ne": setting_id}},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )
    
    update_data["updated_at"] = datetime.utcnow()
    
    result = await ai_settings_coll.find_one_and_update(
        {"setting_id": setting_id},
        {"$set": update_data},
        return_document=True
    )
    
    logger.info(f"Updated AI settings configuration: {setting_id}")
    return AISettings(**result)

@router.delete("/{setting_id}")
async def delete_settings(setting_id: str):
    """Delete an AI settings configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings configuration not found")
    
    if settings.get("is_active"):
        raise HTTPException(status_code=400, detail="Cannot delete active configuration")
    
    await ai_settings_coll.delete_one({"setting_id": setting_id})
    logger.info(f"Deleted AI settings configuration: {setting_id}")
    return {"message": f"Settings configuration {setting_id} deleted successfully"}

@router.post("/{setting_id}/activate", response_model=AISettings)
async def activate_settings(setting_id: str):
    """Activate a specific AI settings configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings configuration not found")
    
    # Deactivate all other configurations
    await ai_settings_coll.update_many(
        {"is_active": True},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
    )
    
    # Activate the specified configuration
    result = await ai_settings_coll.find_one_and_update(
        {"setting_id": setting_id},
        {"$set": {"is_active": True, "updated_at": datetime.utcnow()}},
        return_document=True
    )
    
    logger.info(f"Activated AI settings configuration: {setting_id}")
    return AISettings(**result)

# ==================== MODEL MANAGEMENT APIS ====================

@router.get("/models/available", response_model=List[ModelStatus])
async def list_available_models():
    """List all available AI models and their status"""
    models = []
    
    # Test LLM models
    llm_models = ["llava:7b", "gpt-oss"]
    for model in llm_models:
        status = await _check_model_status(model, ModelType.LLM)
        models.append(status)
    
    # Test embedding models  
    embed_models = ["nomic-embed-text:latest", "all-minilm:l6-v2", "mxbai-embed-large"]
    for model in embed_models:
        status = await _check_model_status(model, ModelType.EMBEDDING)
        models.append(status)
    
    return models

@router.get("/models/{model_name}/status", response_model=ModelStatus)
async def get_model_status(model_name: str, model_type: ModelType):
    """Get the status of a specific model"""
    return await _check_model_status(model_name, model_type)

@router.post("/models/{model_name}/test")
async def test_model(model_name: str, model_type: ModelType, test_input: str = Body(..., embed=True)):
    """Test a specific model with sample input"""
    try:
        if model_type == ModelType.LLM:
            # Test LLM with a simple prompt
            start_time = datetime.utcnow()
            result = ollama_generate(test_input, model=model_name)
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "model_name": model_name,
                "model_type": model_type,
                "response_time_ms": response_time,
                "result": result[:500],  # Truncate long responses
                "test_input": test_input
            }
        
        elif model_type == ModelType.EMBEDDING:
            # Test embedding model
            start_time = datetime.utcnow()
            embedding = ollama_embed(test_input)
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "model_name": model_name,
                "model_type": model_type,
                "response_time_ms": response_time,
                "embedding_dimension": len(embedding) if embedding else 0,
                "test_input": test_input
            }
    
    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "model_type": model_type,
            "error": str(e),
            "test_input": test_input
        }

# ==================== COMPONENT-SPECIFIC CONFIGURATION APIS ====================

@router.get("/{setting_id}/llm", response_model=LLMSettings)
async def get_llm_settings(setting_id: str):
    """Get LLM settings from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return LLMSettings(**settings["llm_settings"])

@router.put("/{setting_id}/llm", response_model=LLMSettings)
async def update_llm_settings(setting_id: str, llm_settings: LLMSettings):
    """Update LLM settings in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"llm_settings": llm_settings.dict(), "updated_at": datetime.utcnow()}}
    )
    return llm_settings

@router.get("/{setting_id}/embedding", response_model=EmbeddingSettings)
async def get_embedding_settings(setting_id: str):
    """Get embedding settings from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return EmbeddingSettings(**settings["embedding_settings"])

@router.put("/{setting_id}/embedding", response_model=EmbeddingSettings)
async def update_embedding_settings(setting_id: str, embedding_settings: EmbeddingSettings):
    """Update embedding settings in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"embedding_settings": embedding_settings.dict(), "updated_at": datetime.utcnow()}}
    )
    return embedding_settings

@router.get("/{setting_id}/thresholds", response_model=ScoringThresholds)
async def get_scoring_thresholds(setting_id: str):
    """Get scoring thresholds from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return ScoringThresholds(**settings["scoring_thresholds"])

@router.put("/{setting_id}/thresholds", response_model=ScoringThresholds)
async def update_scoring_thresholds(setting_id: str, thresholds: ScoringThresholds):
    """Update scoring thresholds in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"scoring_thresholds": thresholds.dict(), "updated_at": datetime.utcnow()}}
    )
    return thresholds

@router.get("/{setting_id}/skills", response_model=SkillWeights)
async def get_skill_weights(setting_id: str):
    """Get skill weights from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return SkillWeights(**settings["skill_weights"])

@router.put("/{setting_id}/skills", response_model=SkillWeights)
async def update_skill_weights(setting_id: str, skill_weights: SkillWeights):
    """Update skill weights in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"skill_weights": skill_weights.dict(), "updated_at": datetime.utcnow()}}
    )
    return skill_weights

@router.get("/{setting_id}/algorithm", response_model=MatchingAlgorithm)
async def get_matching_algorithm(setting_id: str):
    """Get matching algorithm settings from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return MatchingAlgorithm(**settings["matching_algorithm"])

@router.put("/{setting_id}/algorithm", response_model=MatchingAlgorithm)
async def update_matching_algorithm(setting_id: str, algorithm: MatchingAlgorithm):
    """Update matching algorithm settings in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"matching_algorithm": algorithm.dict(), "updated_at": datetime.utcnow()}}
    )
    return algorithm

@router.get("/{setting_id}/prompts", response_model=PromptSettings)
async def get_prompt_settings(setting_id: str):
    """Get prompt settings from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return PromptSettings(**settings["prompt_settings"])

@router.put("/{setting_id}/prompts", response_model=PromptSettings)
async def update_prompt_settings(setting_id: str, prompts: PromptSettings):
    """Update prompt settings in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"prompt_settings": prompts.dict(), "updated_at": datetime.utcnow()}}
    )
    return prompts

@router.get("/{setting_id}/processing", response_model=ProcessingSettings)
async def get_processing_settings(setting_id: str):
    """Get processing settings from a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return ProcessingSettings(**settings["processing_settings"])

@router.put("/{setting_id}/processing", response_model=ProcessingSettings)
async def update_processing_settings(setting_id: str, processing: ProcessingSettings):
    """Update processing settings in a configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    await ai_settings_coll.update_one(
        {"setting_id": setting_id},
        {"$set": {"processing_settings": processing.dict(), "updated_at": datetime.utcnow()}}
    )
    return processing

# ==================== VALIDATION AND TESTING APIS ====================

@router.post("/{setting_id}/validate", response_model=SettingsValidationResult)
async def validate_settings(setting_id: str):
    """Validate an AI settings configuration"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    try:
        ai_settings = AISettings(**settings)
        validation_result = await _validate_ai_settings(ai_settings)
        return validation_result
    except Exception as e:
        return SettingsValidationResult(
            is_valid=False,
            validation_errors=[f"Validation failed: {str(e)}"]
        )

@router.post("/validate", response_model=SettingsValidationResult)
async def validate_settings_payload(settings: AISettings):
    """Validate an AI settings configuration without saving"""
    try:
        validation_result = await _validate_ai_settings(settings)
        return validation_result
    except Exception as e:
        return SettingsValidationResult(
            is_valid=False,
            validation_errors=[f"Validation failed: {str(e)}"]
        )

@router.post("/{setting_id}/clone", response_model=AISettings)
async def clone_settings(setting_id: str, new_name: str = Body(..., embed=True)):
    """Clone an existing AI settings configuration"""
    existing = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not existing:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    # Create new settings with different ID and name
    new_settings = AISettings(**existing)
    new_settings.setting_id = str(uuid.uuid4())
    new_settings.name = new_name
    new_settings.is_active = False
    new_settings.is_default = False
    new_settings.created_at = datetime.utcnow()
    new_settings.updated_at = datetime.utcnow()
    
    await ai_settings_coll.insert_one(new_settings.dict())
    logger.info(f"Cloned AI settings {setting_id} to {new_settings.setting_id}")
    return new_settings

@router.get("/export/{setting_id}")
async def export_settings(setting_id: str):
    """Export AI settings configuration as JSON"""
    settings = await ai_settings_coll.find_one({"setting_id": setting_id})
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    
    # Remove database-specific fields for export
    export_data = {k: v for k, v in settings.items() if k not in ["_id"]}
    
    return JSONResponse(
        content=export_data,
        headers={"Content-Disposition": f"attachment; filename=ai_settings_{setting_id}.json"}
    )

@router.post("/import", response_model=AISettings)
async def import_settings(settings_data: Dict[str, Any]):
    """Import AI settings configuration from JSON"""
    try:
        # Generate new IDs to avoid conflicts
        settings_data["setting_id"] = str(uuid.uuid4())
        settings_data["is_active"] = False
        settings_data["is_default"] = False
        settings_data["created_at"] = datetime.utcnow()
        settings_data["updated_at"] = datetime.utcnow()
        
        settings = AISettings(**settings_data)
        await ai_settings_coll.insert_one(settings.dict())
        
        logger.info(f"Imported AI settings configuration: {settings.setting_id}")
        return settings
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")

# ==================== HELPER FUNCTIONS ====================

async def _check_model_status(model_name: str, model_type: ModelType) -> ModelStatus:
    """Check if a model is available and responding"""
    try:
        start_time = datetime.utcnow()
        
        if model_type == ModelType.LLM:
            # Test with simple prompt
            result = ollama_generate("Hello", model=model_name, temperature=0.1)
            success = bool(result and len(result) > 0)
        else:
            # Test embedding
            result = ollama_embed("test")
            success = bool(result and len(result) > 0)
        
        end_time = datetime.utcnow()
        response_time = (end_time - start_time).total_seconds() * 1000
        
        return ModelStatus(
            model_name=model_name,
            model_type=model_type,
            is_available=success,
            status="available" if success else "error",
            response_time_ms=response_time,
            last_checked=datetime.utcnow()
        )
    
    except Exception as e:
        return ModelStatus(
            model_name=model_name,
            model_type=model_type,
            is_available=False,
            status="error",
            last_checked=datetime.utcnow(),
            error_message=str(e)
        )

async def _validate_ai_settings(settings: AISettings) -> SettingsValidationResult:
    """Validate AI settings configuration"""
    errors = []
    warnings = []
    test_results = {}
    
    # Test LLM model
    try:
        llm_status = await _check_model_status(settings.llm_settings.model_name, ModelType.LLM)
        test_results["llm_test"] = llm_status.dict()
        if not llm_status.is_available:
            errors.append(f"LLM model '{settings.llm_settings.model_name}' is not available")
    except Exception as e:
        errors.append(f"LLM model test failed: {str(e)}")
    
    # Test embedding model
    try:
        embed_status = await _check_model_status(settings.embedding_settings.model_name, ModelType.EMBEDDING)
        test_results["embedding_test"] = embed_status.dict()
        if not embed_status.is_available:
            errors.append(f"Embedding model '{settings.embedding_settings.model_name}' is not available")
    except Exception as e:
        errors.append(f"Embedding model test failed: {str(e)}")
    
    # Validate thresholds
    thresholds = settings.scoring_thresholds
    if thresholds.reject_max >= thresholds.select_min:
        errors.append("reject_max must be less than select_min")
    
    if thresholds.moderate_threshold >= thresholds.strong_threshold:
        warnings.append("moderate_threshold should be less than strong_threshold")
    
    # Validate algorithm weights
    algo = settings.matching_algorithm
    total_weight = algo.embedding_weight + algo.skill_weight + algo.consistency_weight
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Algorithm weights sum to {total_weight}, should sum to 1.0")
    
    return SettingsValidationResult(
        is_valid=len(errors) == 0,
        validation_errors=errors,
        warnings=warnings,
        test_results=test_results
    )