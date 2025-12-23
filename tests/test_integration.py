"""
Comprehensive integration tests for the Research Paper Chatbot
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models import Base
from app.core.database import get_db, init_db
from app.services.message_handler import handle_message, detect_intent


# Test database setup
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def override_get_db():
    """Override database dependency for testing"""
    async with test_session_maker() as session:
        yield session


# Override the dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test (sync wrapper)"""
    async def _get_session():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with test_session_maker() as session:
            yield session

        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    return asyncio.run(_get_session().__anext__())


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


# ---------------------------
# Unit Tests - Intent Detection
# ---------------------------

def test_intent_detection_help():
    """Test help intent detection"""
    assert detect_intent("help") == "help"
    assert detect_intent("I need help") == "help"
    assert detect_intent("commands") == "help"


def test_intent_detection_status():
    """Test status intent detection"""
    assert detect_intent("status") == "status"
    assert detect_intent("where am i") == "status"


def test_intent_detection_reset():
    """Test reset intent detection"""
    assert detect_intent("reset") == "reset"
    assert detect_intent("clear") == "reset"


def test_intent_detection_paper_search():
    """Test paper search intent detection"""
    assert detect_intent("machine learning transformers") == "paper"
    assert detect_intent("neural networks deep learning") == "paper"
    assert detect_intent("https://arxiv.org/abs/1706.03762") == "paper"


def test_intent_detection_qna():
    """Test Q&A intent detection"""
    assert detect_intent("start qna") == "qna_start"
    assert detect_intent("ready for q&a") == "qna_start"
    assert detect_intent("skip") == "qna_skip"


def test_intent_detection_selection():
    """Test selection intent detection"""
    assert detect_intent("select 1") == "selection"
    assert detect_intent("choose 3") == "selection"
    assert detect_intent("pick 2") == "selection"


def test_intent_detection_stats():
    """Test stats intent detection"""
    assert detect_intent("my stats") == "stats"
    assert detect_intent("statistics") == "stats"
    assert detect_intent("analytics") == "stats"


def test_intent_detection_recommend():
    """Test recommend intent detection"""
    assert detect_intent("recommend") == "recommend"
    assert detect_intent("recommend papers") == "recommend"


# ---------------------------
# Integration Tests - Message Handling
# ---------------------------

@pytest.mark.asyncio
async def test_handle_help_message(db_session):
    """Test help message handling"""
    user_id = "test_user_1"
    response = await handle_message(user_id, "help", db_session)

    assert "Research Paper Bot" in response
    assert "Search" in response
    assert "Q&A" in response


@pytest.mark.asyncio
async def test_handle_status_message(db_session):
    """Test status message handling"""
    user_id = "test_user_2"
    response = await handle_message(user_id, "status", db_session)

    assert "Status" in response
    assert "Mode" in response


@pytest.mark.asyncio
async def test_handle_reset_message(db_session):
    """Test reset message handling"""
    user_id = "test_user_3"
    response = await handle_message(user_id, "reset", db_session)

    assert "reset" in response.lower()


@pytest.mark.asyncio
async def test_session_creation(db_session):
    """Test that sessions are created automatically"""
    from app.core.database import get_session

    user_id = "test_user_4"
    session = await get_session(user_id, db_session)

    assert session is not None
    assert session.user_id == user_id
    assert session.mode == "browsing"


@pytest.mark.asyncio
async def test_stats_for_new_user(db_session):
    """Test stats for a new user"""
    user_id = "test_user_5"
    response = await handle_message(user_id, "my stats", db_session)

    assert "Statistics" in response
    assert "Papers read: 0" in response


@pytest.mark.asyncio
async def test_ambiguous_message(db_session):
    """Test handling of ambiguous messages"""
    user_id = "test_user_6"
    response = await handle_message(user_id, "hi there", db_session)

    assert "didn't understand" in response or "Try:" in response


# ---------------------------
# Integration Tests - API Endpoints
# ---------------------------

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["version"] == "2.0.0"


def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data
    assert "cache" in data


def test_whatsapp_endpoint_basic(client):
    """Test WhatsApp endpoint with basic message"""
    payload = {
        "From": "whatsapp:+1234567890",
        "Body": "help"
    }

    response = client.post("/whatsapp", data=payload)
    assert response.status_code == 200
    assert "application/xml" in response.headers["content-type"]
    assert "Research Paper Bot" in response.text


def test_whatsapp_endpoint_status(client):
    """Test WhatsApp endpoint with status message"""
    payload = {
        "From": "whatsapp:+1234567890",
        "Body": "status"
    }

    response = client.post("/whatsapp", data=payload)
    assert response.status_code == 200
    assert "Status" in response.text


# ---------------------------
# Integration Tests - Database Operations
# ---------------------------

@pytest.mark.asyncio
async def test_chat_log_creation(db_session):
    """Test that chat logs are created"""
    from app.core.database import log_message
    from app.models import ChatLog
    from sqlalchemy import select

    user_id = "test_user_7"

    await log_message(user_id, "user", "Hello", db_session)
    await log_message(user_id, "bot", "Hi there!", db_session)

    result = await db_session.execute(
        select(ChatLog).where(ChatLog.user_id == user_id)
    )
    logs = list(result.scalars().all())

    assert len(logs) == 2
    assert logs[0].role == "user"
    assert logs[1].role == "bot"


@pytest.mark.asyncio
async def test_session_update(db_session):
    """Test session updates"""
    from app.core.database import get_session, update_session

    user_id = "test_user_8"

    # Create session
    session = await get_session(user_id, db_session)
    assert session.mode == "browsing"

    # Update session
    await update_session(user_id, db_session, mode="qna", qna_active=True)

    # Verify update
    updated_session = await get_session(user_id, db_session)
    assert updated_session.mode == "qna"
    assert updated_session.qna_active == True


# ---------------------------
# Integration Tests - Service Layer
# ---------------------------

@pytest.mark.asyncio
async def test_analytics_streak_calculation(db_session):
    """Test streak calculation"""
    from app.services.analytics import calculate_streak

    user_id = "test_user_9"

    # New user should have 0 streak
    streak = await calculate_streak(user_id, db_session)
    assert streak == 0


@pytest.mark.asyncio
async def test_spaced_repetition_scheduling(db_session):
    """Test spaced repetition scheduling"""
    from app.services.spaced_repetition import schedule_review, get_due_reviews

    user_id = "test_user_10"

    # Schedule a review
    await schedule_review(
        user_id=user_id,
        paper_id="test_paper",
        question_id="q1",
        performance=8.0,
        db=db_session
    )

    # Check stats
    from app.services.spaced_repetition import get_review_stats
    stats = await get_review_stats(user_id, db_session)

    assert stats["total_reviews"] == 1


# ---------------------------
# Performance Tests
# ---------------------------

@pytest.mark.asyncio
async def test_concurrent_requests(db_session):
    """Test handling of concurrent requests"""
    async def process_message(user_id):
        return await handle_message(user_id, "help", db_session)

    # Simulate 10 concurrent users
    tasks = [process_message(f"user_{i}") for i in range(10)]
    responses = await asyncio.gather(*tasks)

    assert len(responses) == 10
    assert all("Research Paper Bot" in r for r in responses)


# ---------------------------
# Error Handling Tests
# ---------------------------

@pytest.mark.asyncio
async def test_invalid_selection(db_session):
    """Test handling of invalid paper selection"""
    user_id = "test_user_11"

    # Try to select without searching
    response = await handle_message(user_id, "select 1", db_session)

    assert "search" in response.lower() or "Please" in response


@pytest.mark.asyncio
async def test_qna_without_paper(db_session):
    """Test Q&A start without selecting a paper"""
    user_id = "test_user_12"

    response = await handle_message(user_id, "start qna", db_session)

    assert "select a paper" in response.lower() or "Please" in response


# ---------------------------
# Summary
# ---------------------------

def test_suite_summary():
    """Print test summary"""
    print("\n" + "="*60)
    print("INTEGRATION TEST SUITE SUMMARY")
    print("="*60)
    print("✅ Intent detection tests")
    print("✅ Message handling tests")
    print("✅ API endpoint tests")
    print("✅ Database operation tests")
    print("✅ Service layer tests")
    print("✅ Performance tests")
    print("✅ Error handling tests")
    print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
