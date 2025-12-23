"""
Configuration management for Research Paper Chatbot
"""

import os
from typing import Optional
from dotenv import load_dotenv
import logging

load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Settings:
    """Application settings"""

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./whatsapp_bot_async.db"
    )

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # AI Models
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.5"))

    # Twilio
    TWILIO_ACCOUNT_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_WHATSAPP_FROM: Optional[str] = os.getenv("TWILIO_WHATSAPP_FROM")

    # Server
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Features
    ENABLE_PDF_PROCESSING: bool = os.getenv("ENABLE_PDF_PROCESSING", "true").lower() == "true"
    ENABLE_VOICE_MESSAGES: bool = os.getenv("ENABLE_VOICE_MESSAGES", "true").lower() == "true"
    ENABLE_FIGURES: bool = os.getenv("ENABLE_FIGURES", "true").lower() == "true"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # Cache TTL (seconds)
    CACHE_TTL_SEARCH: int = 3600  # 1 hour
    CACHE_TTL_SUMMARY: Optional[int] = None  # Permanent
    CACHE_TTL_SESSION: int = 86400  # 24 hours

    def validate(self, strict: bool = False):
        """Validate required settings"""
        if not self.TWILIO_ACCOUNT_SID or not self.TWILIO_AUTH_TOKEN:
            message = (
                "Missing required Twilio credentials. "
                "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env"
            )
            if strict:
                raise RuntimeError(message)
            else:
                logger.warning(message)

        if not self.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set. AI features will be limited.")


# Global settings instance
settings = Settings()
