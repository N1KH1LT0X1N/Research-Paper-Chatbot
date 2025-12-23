"""
Comprehensive tests for async_research_bot
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Import from main app
import sys
sys.path.insert(0, '..')
from async_research_bot import (
    app, Base, Session as SessionModel, Paper, UserHistory,
    detect_intent, parse_paper_sections, compact_summary,
    generate_qna_items, evaluate_answer_semantic,
    init_db, get_session, update_session
)


# Test database
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def test_db():
    """Create test database"""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    yield async_session_maker

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def http_client():
    """HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# ---------------------------
# Intent Detection Tests
# ---------------------------

class TestIntentDetection:
    """Test intent detection"""

    def test_qna_start_intents(self):
        assert detect_intent("ready for Q&A") == "qna_start"
        assert detect_intent("let's do qna") == "qna_start"
        assert detect_intent("start qna") == "qna_start"

    def test_command_intents(self):
        assert detect_intent("help") == "help"
        assert detect_intent("status") == "status"
        assert detect_intent("reset") == "reset"

    def test_feature_intents(self):
        assert detect_intent("show figures") == "show_figures"
        assert detect_intent("show citations") == "show_citations"
        assert detect_intent("export") == "export"
        assert detect_intent("add to list") == "add_to_list"
        assert detect_intent("my stats") == "my_stats"
        assert detect_intent("recommend") == "recommend"

    def test_selection_intent(self):
        assert detect_intent("select 1") == "selection"
        assert detect_intent("choose 2") == "selection"
        assert detect_intent("pick 3") == "selection"

    def test_paper_search_intents(self):
        assert detect_intent("transformer attention mechanisms") == "paper"
        assert detect_intent("https://arxiv.org/abs/1706.03762") == "paper"
        assert detect_intent("10.48550/arXiv.1706.03762") == "paper"

    def test_ambiguous_intent(self):
        assert detect_intent("hi") == "ambiguous"
        assert detect_intent("ok") == "ambiguous"


# ---------------------------
# PDF Processing Tests
# ---------------------------

class TestPDFProcessing:
    """Test PDF text extraction and parsing"""

    def test_parse_paper_sections(self):
        sample_text = """
        Abstract
        This is the abstract of the paper.

        1. Introduction
        This is the introduction section with important context.

        2. Methods
        We propose a new method for solving this problem.

        3. Results
        Our experiments show significant improvements.

        4. Conclusion
        We conclude that our approach is effective.

        References
        [1] Smith et al. 2020
        """

        sections = parse_paper_sections(sample_text)

        assert "Introduction" in sections or "Abstract" in sections
        # Should extract at least some sections
        assert len(sections) > 0


# ---------------------------
# Summarization Tests
# ---------------------------

class TestSummarization:
    """Test summary generation and formatting"""

    def test_compact_summary(self):
        structured_text = """
## Introduction
This is a test introduction with some content.

## Methodology
This is the methodology section.

## Results
These are the results.

## Conclusions
These are the conclusions.
"""

        result = compact_summary(
            title="Test Paper",
            year=2023,
            authors="Smith, Jones",
            url="https://example.com",
            structured_text=structured_text,
            limit=1400
        )

        assert "Test Paper" in result
        assert "Smith, Jones" in result
        assert len(result) <= 1400  # Should respect limit
        assert "*Introduction*" in result or "*Methodology*" in result


# ---------------------------
# Q&A Tests
# ---------------------------

class TestQnA:
    """Test Q&A generation and evaluation"""

    @pytest.mark.asyncio
    async def test_generate_qna_items(self):
        title = "Attention Is All You Need"
        source_text = "We propose the Transformer, a novel architecture based solely on attention mechanisms."

        items = await generate_qna_items(title, source_text, difficulty="medium", n=3)

        assert len(items) <= 3
        assert all('q' in item for item in items)
        assert all('a_keywords' in item for item in items)

    @pytest.mark.asyncio
    async def test_evaluate_answer_semantic(self):
        question = "What is the main contribution?"
        user_answer = "The Transformer architecture using attention mechanisms"
        reference = "The paper introduces the Transformer, which relies on attention."
        keywords = ["transformer", "attention"]

        score, feedback = await evaluate_answer_semantic(
            question, user_answer, reference, keywords
        )

        assert 0 <= score <= 10
        assert isinstance(feedback, str)
        assert len(feedback) > 0


# ---------------------------
# Database Tests
# ---------------------------

class TestDatabase:
    """Test database operations"""

    @pytest.mark.asyncio
    async def test_create_session(self, test_db):
        async with test_db() as db:
            # Create session
            session = await get_session("whatsapp:+1234567890", db)

            assert session is not None
            assert session.user_id == "whatsapp:+1234567890"
            assert session.mode == "browsing"

    @pytest.mark.asyncio
    async def test_update_session(self, test_db):
        async with test_db() as db:
            user_id = "whatsapp:+1234567890"

            # Create session
            session = await get_session(user_id, db)

            # Update
            await update_session(user_id, db, mode="qna", score=10)

            # Retrieve again
            from sqlalchemy import select
            result = await db.execute(select(SessionModel).where(SessionModel.user_id == user_id))
            updated_session = result.scalar_one()

            assert updated_session.mode == "qna"
            assert updated_session.score == 10

    @pytest.mark.asyncio
    async def test_store_paper(self, test_db):
        async with test_db() as db:
            paper_data = {
                "paperId": "test123",
                "title": "Test Paper",
                "authors": "Author One, Author Two",
                "year": 2023,
                "abstract": "This is a test abstract",
                "url": "https://example.com",
                "citationCount": 100
            }

            from async_research_bot import store_paper
            paper = await store_paper(paper_data, db)

            assert paper.paper_id == "test123"
            assert paper.title == "Test Paper"
            assert paper.citation_count == 100


# ---------------------------
# API Endpoint Tests
# ---------------------------

class TestEndpoints:
    """Test FastAPI endpoints"""

    @pytest.mark.asyncio
    async def test_root_endpoint(self, http_client):
        response = await http_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_client):
        response = await http_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data


# ---------------------------
# Search Tests
# ---------------------------

class TestPaperSearch:
    """Test paper search functionality"""

    @pytest.mark.asyncio
    async def test_search_semantic_scholar(self):
        from async_research_bot import search_papers_semantic_scholar

        # Note: This makes real API calls, might want to mock in production
        results = await search_papers_semantic_scholar("transformer", limit=3)

        # May return empty if API is down, but should not crash
        assert isinstance(results, list)
        if results:
            assert "title" in results[0]
            assert "paperId" in results[0]

    @pytest.mark.asyncio
    async def test_search_arxiv(self):
        from async_research_bot import search_papers_arxiv

        results = await search_papers_arxiv("neural networks", limit=3)

        assert isinstance(results, list)
        if results:
            assert "title" in results[0]


# ---------------------------
# Utility Tests
# ---------------------------

class TestUtilities:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_calculate_streak(self, test_db):
        from async_research_bot import calculate_streak
        from datetime import datetime, timedelta

        async with test_db() as db:
            user_id = "whatsapp:+1234567890"

            # Add history for 3 consecutive days
            for i in range(3):
                day = datetime.utcnow() - timedelta(days=i)
                history = UserHistory(
                    user_id=user_id,
                    paper_id="test_paper",
                    action="read",
                    timestamp=day
                )
                db.add(history)

            await db.commit()

            # Calculate streak
            streak = await calculate_streak(user_id, db)

            assert streak >= 1  # At least 1 day streak


# ---------------------------
# Integration Tests
# ---------------------------

class TestIntegration:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_browsing_flow(self, test_db):
        """Test: Search -> Select -> Summary"""
        from async_research_bot import handle_browsing

        async with test_db() as db:
            user_id = "whatsapp:+1234567890"

            # Step 1: Search
            response1 = await handle_browsing(user_id, "machine learning", db)
            assert "Search Results" in response1 or "papers" in response1.lower()

            # Step 2: Select (this might fail without real results, but shouldn't crash)
            response2 = await handle_browsing(user_id, "select 1", db)
            # Should either show summary or say no results
            assert isinstance(response2, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
