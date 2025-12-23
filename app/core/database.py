"""
Database connection and session management
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update as sql_update
from datetime import datetime
from typing import Optional, Any

from .config import settings, logger
from app.models import Base, Session as SessionModel


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=20,
    max_overflow=10
)

# Create session maker
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def get_db():
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def get_session(user_id: str, db: AsyncSession) -> SessionModel:
    """Get or create user session"""
    result = await db.execute(
        select(SessionModel).where(SessionModel.user_id == user_id)
    )
    session = result.scalar_one_or_none()

    if not session:
        session = SessionModel(
            user_id=user_id,
            updated_at=datetime.utcnow()
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)

    return session


async def update_session(user_id: str, db: AsyncSession, **kwargs):
    """Update session attributes"""
    if not kwargs:
        return

    kwargs['updated_at'] = datetime.utcnow()

    await db.execute(
        sql_update(SessionModel)
        .where(SessionModel.user_id == user_id)
        .values(**kwargs)
    )
    await db.commit()


async def log_message(user_id: str, role: str, message: str, db: AsyncSession):
    """Log a chat message"""
    from app.models import ChatLog

    log = ChatLog(
        user_id=user_id,
        role=role,
        message=message,
        created_at=datetime.utcnow()
    )
    db.add(log)
    await db.commit()


__all__ = [
    "engine",
    "async_session_maker",
    "init_db",
    "get_db",
    "get_session",
    "update_session",
    "log_message"
]
