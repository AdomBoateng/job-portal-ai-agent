import motor.motor_asyncio
from pymongo import ASCENDING
import os
from dotenv import load_dotenv

# Import logging
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Load environment variables from .env
load_dotenv()

# Connection string (from .env or fallback)
MONGO_DETAILS = os.getenv(
    "MONGO_DETAILS",
    "mongodb+srv://gideonboateng898_db_user:zdu3nKvC2VZUC4tR@cluster0.qcza2iy.mongodb.net/?retryWrites=true&w=majority"
)

DB_NAME = os.getenv("DB_NAME", "AI_agent_db")

logger.info(f"Initializing MongoDB connection to database: {DB_NAME}")

# Initialize client
try:
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
    db = client[DB_NAME]
    logger.info("MongoDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB client: {e}")
    raise

# Collections
jds_coll = db["jds"]
cvs_coll = db["cvs"]
sessions_coll = db["sessions"]
reports_coll = db["reports"]
ai_settings_coll = db["ai_settings"]


async def init_indexes():
    """Index initialization for collections."""
    logger.info("Starting database index initialization")
    
    try:
        # JDs collection - compound unique index to allow same jd_id across different sessions
        try:
            # First try to drop the old single-field unique index if it exists
            try:
                await jds_coll.drop_index("jd_id_1")
                logger.debug("Dropped old unique index on jds.jd_id")
            except Exception:
                pass  # Index may not exist
            
            # Create compound unique index on (jd_id, session_id)
            await jds_coll.create_index([("jd_id", ASCENDING), ("session_id", ASCENDING)], unique=True)
            logger.debug("Created compound unique index on jds.(jd_id, session_id)")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Compound index on jds.(jd_id, session_id) already exists")
            else:
                logger.warning(f"Could not create compound unique index on jds.(jd_id, session_id): {e}")
        
        # CVs collection - compound unique index to allow same cv_id across different sessions
        try:
            # First try to drop the old single-field unique index if it exists
            try:
                await cvs_coll.drop_index("cv_id_1")
                logger.debug("Dropped old unique index on cvs.cv_id")
            except Exception:
                pass  # Index may not exist
            
            # Create compound unique index on (cv_id, session_id)
            await cvs_coll.create_index([("cv_id", ASCENDING), ("session_id", ASCENDING)], unique=True)
            logger.debug("Created compound unique index on cvs.(cv_id, session_id)")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Compound index on cvs.(cv_id, session_id) already exists")
            else:
                logger.warning(f"Could not create compound unique index on cvs.(cv_id, session_id): {e}")
        
        # Sessions collection - handle potential duplicates
        try:
            await sessions_coll.create_index([("session_id", ASCENDING)], unique=True)
            logger.debug("Created unique index on sessions.session_id")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Index on sessions.session_id already exists")
            else:
                logger.warning(f"Could not create unique index on sessions.session_id: {e}")
        
        # Reports collection
        try:
            await reports_coll.create_index([("match_report_id", ASCENDING)], unique=True)
            logger.debug("Created unique index on reports.match_report_id")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Index on reports.match_report_id already exists")
            else:
                logger.warning(f"Could not create unique index on reports.match_report_id: {e}")

        # AI Settings indexes
        try:
            await ai_settings_coll.create_index([("setting_id", ASCENDING)], unique=True)
            logger.debug("Created unique index on ai_settings.setting_id")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug("Index on ai_settings.setting_id already exists")
            else:
                logger.warning(f"Could not create unique index on ai_settings.setting_id: {e}")
        
        # Non-unique indexes for ai_settings (these are safer)
        try:
            await ai_settings_coll.create_index([("name", ASCENDING)])
            await ai_settings_coll.create_index([("is_active", ASCENDING)])
            await ai_settings_coll.create_index([("is_default", ASCENDING)])
            await ai_settings_coll.create_index([("created_at", ASCENDING)])
            await ai_settings_coll.create_index([("tags", ASCENDING)])
            logger.debug("Created additional indexes on ai_settings collection")
        except Exception as e:
            logger.warning(f"Could not create some ai_settings indexes: {e}")
        
        logger.info("Database index initialization completed successfully")
        
    except Exception as e:
        logger.warning(f"Database index initialization had issues: {e}")
        logger.info("Application will continue without all indexes - some operations may be slower")

def to_dict(doc):
    if not doc:
        return None
    doc["_id"] = str(doc["_id"])
    return doc