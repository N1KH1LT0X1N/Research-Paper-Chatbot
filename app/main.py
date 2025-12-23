"""
Research Paper Chatbot - Main Application
FastAPI async application with WhatsApp integration
"""

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
import uvicorn

from app.core.config import settings, logger
from app.core.database import init_db, get_db, log_message
from app.core.cache import cache
from app.services.message_handler import handle_message

# Validate settings on startup
settings.validate()

# Initialize Twilio (optional - only if credentials are set)
twilio_client = None
twilio_validator = None

if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
    twilio_client = TwilioClient(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    twilio_validator = RequestValidator(settings.TWILIO_AUTH_TOKEN)
    logger.info("Twilio client initialized")
else:
    logger.warning("Twilio not configured - webhook signature verification disabled")

# Create FastAPI app
app = FastAPI(
    title="Research Paper Chatbot",
    version="2.0.0",
    description="AI-powered research assistant via WhatsApp"
)


# ---------------------------
# Startup & Shutdown Events
# ---------------------------

@app.on_event("startup")
async def startup():
    """Initialize application on startup"""
    logger.info("Starting Research Paper Chatbot v2.0")

    # Initialize database
    await init_db()

    # Connect to cache
    await cache.connect()

    logger.info("Application ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")

    # Close cache connection
    await cache.disconnect()

    logger.info("Shutdown complete")


# ---------------------------
# Helper Functions
# ---------------------------

async def verify_twilio_signature(request: Request) -> bool:
    """Verify Twilio webhook signature"""
    if not twilio_validator:
        logger.warning("Twilio validator not configured - skipping signature verification")
        return True  # Allow in development mode

    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)
    form_data = await request.form()
    params = dict(form_data)

    return twilio_validator.validate(url, params, signature)


def split_message(text: str, limit: int = 1500) -> list[str]:
    """Split message into chunks with word boundaries"""
    chunks = []

    while len(text) > limit:
        # Find last space before limit
        split_pos = text.rfind(' ', 0, limit)
        if split_pos == -1:
            split_pos = limit

        chunks.append(text[:split_pos])
        text = text[split_pos:].lstrip()

    if text:
        chunks.append(text)

    # Add part indicators if multiple chunks
    if len(chunks) > 1:
        tagged = []
        for i, chunk in enumerate(chunks, 1):
            tagged.append(f"({i}/{len(chunks)})\n{chunk}")
        return tagged

    return chunks


# Message handler is now in app.services.message_handler


# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "version": "2.0.0",
        "message": "Research Paper Chatbot API"
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "cache": "connected" if cache.redis else "local",
        "twilio": "configured" if settings.TWILIO_ACCOUNT_SID else "not configured"
    }


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle incoming WhatsApp messages"""

    # Verify signature (disabled for testing, enable in production)
    # if not await verify_twilio_signature(request):
    #     raise HTTPException(status_code=403, detail="Invalid signature")

    # Parse request
    form_data = await request.form()
    user_id = form_data.get("From", "unknown")
    body = form_data.get("Body", "").strip()

    logger.info(f"Message from {user_id}: {body}")

    # Log incoming message
    await log_message(user_id, "user", body, db)

    # Handle message
    reply_text = await handle_message(user_id, body, db)

    # Log response
    await log_message(user_id, "bot", reply_text, db)

    # Build TwiML response
    resp = MessagingResponse()
    for chunk in split_message(reply_text):
        resp.message(chunk)

    return Response(content=str(resp), media_type="application/xml")


# ---------------------------
# Main Entry Point
# ---------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False
    )
