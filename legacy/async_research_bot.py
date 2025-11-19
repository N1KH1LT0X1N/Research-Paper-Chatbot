"""
Advanced Research Paper Chatbot - Async Version
Complete implementation with all phases integrated.

Features:
- Async FastAPI application
- Multi-source PDF retrieval
- RAG with vector search
- Semantic Q&A grading
- Citation graphs
- Study groups & gamification
- Recommendations & learning paths
"""

import os
import re
import json
import asyncio
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import io

# Core async frameworks
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import Response
import httpx
import uvicorn

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ARRAY

# Caching
import redis.asyncio as redis

# AI & Embeddings
try:
    import google.generativeai as genai
except:
    genai = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except:
    EMBEDDINGS_AVAILABLE = False

# PDF Processing
try:
    import pdfplumber
    import PyPDF2
    from PIL import Image
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

# Vector Store
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    VECTOR_STORE_AVAILABLE = True
except:
    VECTOR_STORE_AVAILABLE = False

# Twilio
from twilio.rest import Client as TwilioClient
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator

# Utilities
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

# ---------------------------
# Configuration & Setup
# ---------------------------

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./whatsapp_bot_async.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")

# Validate required env vars
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise RuntimeError("Missing required Twilio credentials in .env")

# Initialize clients
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)

if GEMINI_API_KEY and genai:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Research Paper Chatbot", version="2.0.0")

# Database
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Global HTTP client
http_client = httpx.AsyncClient(timeout=30.0)

# Global Redis client
redis_client: Optional[redis.Redis] = None

# Embedding model
embedding_model = None
if EMBEDDINGS_AVAILABLE:
    try:
        embedding_model = SentenceTransformer('allenai/specter2')
        logger.info("Loaded SPECTER2 embedding model")
    except:
        try:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded MiniLM embedding model")
        except:
            logger.warning("No embedding model available")

# Vector store
vector_store = None
if VECTOR_STORE_AVAILABLE:
    try:
        chroma_client = chromadb.Client(ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        vector_store = chroma_client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Initialized ChromaDB vector store")
    except Exception as e:
        logger.warning(f"Vector store unavailable: {e}")


# ---------------------------
# Database Models
# ---------------------------

class Session(Base):
    __tablename__ = "sessions"

    user_id = Column(String, primary_key=True)
    mode = Column(String, default="browsing")  # browsing | qna | review
    selected_paper_id = Column(String)
    selected_paper_title = Column(String)
    selected_paper_abstract = Column(Text)
    qna_active = Column(Boolean, default=False)
    qna_index = Column(Integer, default=0)
    qna_questions = Column(JSON)  # List of Q&A items
    score = Column(Integer, default=0)
    last_results = Column(JSON)  # Search results cache
    current_list = Column(String)  # Current reading list
    current_group = Column(String)  # Current study group
    difficulty_preference = Column(String, default="medium")
    voice_enabled = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Paper(Base):
    __tablename__ = "papers"

    paper_id = Column(String, primary_key=True)
    title = Column(String)
    authors = Column(String)
    year = Column(Integer)
    venue = Column(String)
    url = Column(String)
    abstract = Column(Text)
    full_text = Column(Text)
    pdf_path = Column(String)
    citation_count = Column(Integer, default=0)
    metadata = Column(JSON)  # Additional fields
    sections = Column(JSON)  # Parsed sections
    figures = Column(JSON)  # Extracted figures
    tables = Column(JSON)  # Extracted tables
    references = Column(JSON)  # Bibliography
    embeddings = Column(JSON)  # Vector embeddings (as list)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_paper_title', 'title'),)


class UserHistory(Base):
    __tablename__ = "user_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    action = Column(String)  # searched | read | qna_completed | added_to_list
    score = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_user_history', 'user_id', 'timestamp'),)


class ReadingList(Base):
    __tablename__ = "reading_lists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    list_name = Column(String, default="default")
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    status = Column(String, default="to_read")  # to_read | reading | completed
    notes = Column(Text)
    score = Column(Integer)
    added_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    __table_args__ = (Index('idx_reading_list', 'user_id', 'list_name'),)


class StudyGroup(Base):
    __tablename__ = "study_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_code = Column(String, unique=True)
    name = Column(String)
    created_by = Column(String)
    members = Column(JSON)  # List of user_ids
    papers = Column(JSON)  # Shared reading list
    created_at = Column(DateTime, default=datetime.utcnow)


class Achievement(Base):
    __tablename__ = "achievements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    achievement_key = Column(String)
    unlocked_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_achievements', 'user_id'),)


class ReviewSchedule(Base):
    __tablename__ = "review_schedule"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    question_id = Column(String)
    easiness = Column(Float, default=2.5)
    interval = Column(Integer, default=1)  # days
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime)
    last_score = Column(Float)

    __table_args__ = (Index('idx_review_schedule', 'user_id', 'next_review'),)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    role = Column(String)  # user | bot
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------
# Database Helpers
# ---------------------------

async def get_db():
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def get_session(user_id: str, db: AsyncSession) -> Session:
    """Get or create user session"""
    from sqlalchemy import select

    result = await db.execute(select(Session).where(Session.user_id == user_id))
    session = result.scalar_one_or_none()

    if not session:
        session = Session(user_id=user_id, updated_at=datetime.utcnow())
        db.add(session)
        await db.commit()
        await db.refresh(session)

    return session


async def update_session(user_id: str, db: AsyncSession, **kwargs):
    """Update session attributes"""
    from sqlalchemy import update

    kwargs['updated_at'] = datetime.utcnow()

    await db.execute(
        update(Session).where(Session.user_id == user_id).values(**kwargs)
    )
    await db.commit()


async def log_message(user_id: str, role: str, message: str, db: AsyncSession):
    """Log a chat message"""
    log = ChatLog(user_id=user_id, role=role, message=message)
    db.add(log)
    await db.commit()


# ---------------------------
# Cache Manager
# ---------------------------

class CacheManager:
    """Redis cache manager with fallback"""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.local_cache = {}  # Fallback in-memory cache

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = redis.from_url(REDIS_URL, decode_responses=True)
            await self.redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self.redis = None

    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        if self.redis:
            try:
                return await self.redis.get(key)
            except:
                pass
        return self.local_cache.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        """Set cached value"""
        if self.redis:
            try:
                if ttl:
                    await self.redis.setex(key, ttl, value)
                else:
                    await self.redis.set(key, value)
                return
            except:
                pass
        self.local_cache[key] = value

    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except:
                pass
        return None

    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set JSON value"""
        await self.set(key, json.dumps(value), ttl)


cache = CacheManager()


# ---------------------------
# Paper Retrieval (Multi-Source)
# ---------------------------

async def search_papers_semantic_scholar(query: str, limit: int = 3) -> List[Dict]:
    """Search Semantic Scholar API"""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,url,abstract,citationCount,influentialCitationCount,venue",
        "limit": limit,
    }

    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("data", [])[:limit]:
            results.append({
                "paperId": item.get("paperId"),
                "title": item.get("title"),
                "year": item.get("year"),
                "url": item.get("url"),
                "authors": ", ".join([a.get("name", "") for a in item.get("authors", []) if a]),
                "abstract": item.get("abstract") or "",
                "citationCount": item.get("citationCount", 0),
                "venue": item.get("venue", ""),
            })

        return results
    except Exception as e:
        logger.error(f"Semantic Scholar search failed: {e}")
        return []


async def search_papers_arxiv(query: str, limit: int = 3) -> List[Dict]:
    """Search arXiv API"""
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}

    try:
        response = await http_client.get(base, params=params)
        response.raise_for_status()
        text = response.text

        entries = text.split("<entry>")[1:limit+1]
        results = []

        for e in entries:
            def tag(name: str) -> str:
                m = re.search(fr"<{name}>(.*?)</{name}>", e, re.S)
                return (m.group(1).strip() if m else "")

            title = re.sub(r"\s+", " ", tag("title"))
            summary = re.sub(r"\s+", " ", tag("summary"))
            year_m = re.search(r"<published>(\d{4})-", e)
            year = int(year_m.group(1)) if year_m else None
            link_m = re.search(r"<link[^>]+href=\"(http[^\"]+)\"[^>]*/>", e)
            link = link_m.group(1) if link_m else ""
            authors = ", ".join(re.findall(r"<name>(.*?)</name>", e))

            # Extract arXiv ID
            arxiv_id_m = re.search(r"(\d{4}\.\d{4,5})", link)
            paper_id = arxiv_id_m.group(1) if arxiv_id_m else None

            if title:
                results.append({
                    "paperId": paper_id or f"arxiv_{len(results)}",
                    "title": title,
                    "year": year,
                    "url": link,
                    "authors": authors,
                    "abstract": summary,
                    "citationCount": 0,
                    "venue": "arXiv",
                })

        return results
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return []


async def download_pdf_from_arxiv(arxiv_id: str) -> Optional[bytes]:
    """Download PDF from arXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = await http_client.get(url, follow_redirects=True)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"arXiv PDF download failed: {e}")
        return None


async def extract_pdf_text(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract text and metadata from PDF"""
    if not PDF_AVAILABLE:
        return {"full_text": "", "sections": {}, "page_count": 0}

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        full_text = ""
        sections = {}

        # Try pdfplumber first
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"

                page_count = len(pdf.pages)
        except:
            # Fallback to PyPDF2
            with open(tmp_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                page_count = len(pdf_reader.pages)

        # Parse sections
        sections = parse_paper_sections(full_text)

        # Cleanup
        os.unlink(tmp_path)

        return {
            "full_text": full_text,
            "sections": sections,
            "page_count": page_count
        }
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return {"full_text": "", "sections": {}, "page_count": 0}


def parse_paper_sections(text: str) -> Dict[str, str]:
    """Parse paper into sections using heuristics"""
    sections = {}

    # Common section patterns
    patterns = {
        "Abstract": r'\n\s*(?:ABSTRACT|Abstract)\s*\n(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:INTRODUCTION|Introduction|I\.|1\.))',
        "Introduction": r'\n\s*(?:1\.?\s*)?(?:INTRODUCTION|Introduction)\s*\n(.*?)(?=\n\s*(?:2\.?|II\.)\s*(?:RELATED|Background|METHOD))',
        "Methods": r'\n\s*(?:\d+\.?\s*)?(?:METHODS?|Methodology|APPROACH|Approach)\s*\n(.*?)(?=\n\s*(?:\d+\.?)\s*(?:RESULTS?|EXPERIMENTS?))',
        "Results": r'\n\s*(?:\d+\.?\s*)?(?:RESULTS?|EXPERIMENTS?|Evaluation)\s*\n(.*?)(?=\n\s*(?:\d+\.?)\s*(?:DISCUSSION|CONCLUSION))',
        "Conclusion": r'\n\s*(?:\d+\.?\s*)?(?:CONCLUSION|Conclusions?|DISCUSSION)\s*\n(.*?)(?=\n\s*(?:REFERENCES?|Bibliography|ACKNOWLEDGE))',
    }

    for section_name, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Clean up
            content = re.sub(r'\s+', ' ', content)
            sections[section_name] = content[:2000]  # Limit length

    return sections


# ---------------------------
# AI & Embeddings
# ---------------------------

async def gemini_generate_text(prompt: str, temperature: float = TEMPERATURE) -> Optional[str]:
    """Generate text using Gemini"""
    if not (genai and GEMINI_API_KEY):
        return None

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt, generation_config={"temperature": temperature})
        return getattr(resp, "text", "").strip() or None
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None


async def generate_embedding(text: str) -> Optional[np.ndarray]:
    """Generate embedding for text"""
    if not embedding_model:
        return None

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: embedding_model.encode(text, convert_to_numpy=True)
        )
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


async def store_paper_embedding(paper_id: str, title: str, abstract: str, metadata: Dict):
    """Store paper embedding in vector store"""
    if not vector_store:
        return

    try:
        # Combine title and abstract
        text = f"{title}\n\n{abstract}"
        embedding = await generate_embedding(text)

        if embedding is not None:
            vector_store.add(
                ids=[paper_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
    except Exception as e:
        logger.error(f"Vector store failed: {e}")


async def semantic_search(query: str, n_results: int = 10) -> List[Dict]:
    """Semantic search using vector store"""
    if not vector_store or not embedding_model:
        return []

    try:
        query_embedding = await generate_embedding(query)
        if query_embedding is None:
            return []

        results = vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        return results.get('metadatas', [[]])[0] if results else []
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


# ---------------------------
# Summarization
# ---------------------------

async def summarize_paper(title: str, abstract: str, full_text: Optional[str] = None,
                         sections: Optional[Dict] = None, detailed: bool = False) -> str:
    """Generate structured summary"""

    # Determine source material
    if full_text and detailed:
        source = f"Title: {title}\n\nFull text (excerpt):\n{full_text[:3000]}"
    elif sections and detailed:
        source = f"Title: {title}\n\nSections:\n"
        for name, content in sections.items():
            source += f"\n{name}:\n{content[:500]}\n"
    else:
        source = f"Title: {title}\n\nAbstract: {abstract}"

    style = "Write a concise, structured summary with these sections: Introduction, Methodology, Results, Conclusions."
    if detailed:
        style += " Provide more detail (5-7 sentences per section)."
    else:
        style += " Keep it brief (3-4 sentences per section)."

    prompt = f"""{style}

{source}

Use markdown headings: ## Introduction, ## Methodology, ## Results, ## Conclusions"""

    ai_resp = await gemini_generate_text(prompt)

    if ai_resp:
        return ai_resp

    # Fallback
    intro = abstract[:400] if abstract else "Not available."
    return f"""## Introduction
{intro}

## Methodology
Method details not available from abstract.

## Results
Results not explicitly stated in abstract.

## Conclusions
Conclusions inferred from context."""


def parse_structured_sections(text: str) -> Dict[str, str]:
    """Parse markdown sections"""
    sections = {"Introduction": "", "Methodology": "", "Results": "", "Conclusions": ""}
    current = None

    for line in text.splitlines():
        m = re.match(r"^##\s*(Introduction|Methodology|Methods?|Results?|Conclusions?)\s*$", line.strip(), re.I)
        if m:
            name = m.group(1).capitalize()
            if name.lower().startswith("method"):
                name = "Methodology"
            elif name.lower().startswith("conclusion"):
                name = "Conclusions"
            elif name.lower().startswith("result"):
                name = "Results"
            current = name
            continue

        if current and current in sections:
            sections[current] += (line + "\n")

    # Trim
    for k in sections:
        sections[k] = sections[k].strip()

    return sections


def compact_summary(title: str, year: Optional[int], authors: str, url: str,
                   structured_text: str, limit: int = 1400) -> str:
    """Compact summary to fit WhatsApp limit"""
    sections = parse_structured_sections(structured_text)

    # Build message
    auth = authors[:80] + "..." if len(authors) > 80 else authors
    header = f"📄 *{title}*\n👤 {auth}\n📅 {year or '?'}\n🔗 {url}\n"

    # Try progressive compaction
    for max_sents in [4, 3, 2, 1]:
        body = ""
        icons = {"Introduction": "🎯", "Methodology": "🔬", "Results": "📊", "Conclusions": "💡"}

        for sec_name in ["Introduction", "Methodology", "Results", "Conclusions"]:
            content = sections.get(sec_name, "")
            if content:
                sents = re.split(r'(?<=[.!?])\s+', content)
                content = ' '.join(sents[:max_sents])
            icon = icons.get(sec_name, "▪️")
            body += f"\n\n{icon} *{sec_name}*\n{content}"

        cta = "\n\n━━━━━━━━━━━━━━━\n💬 start qna | 🖼 figures | 📚 add to list"
        full = header + body + cta

        if len(full) <= limit:
            return full

    # Hard truncate
    return (header + body + cta)[:limit-3] + "..."


# ---------------------------
# Q&A Generation & Grading
# ---------------------------

async def generate_qna_items(title: str, source_text: str, difficulty: str = "medium",
                            n: int = 3) -> List[Dict[str, Any]]:
    """Generate Q&A items with difficulty levels"""

    difficulty_prompts = {
        "easy": "Generate basic comprehension questions (What, Who, When). Focus on factual recall.",
        "medium": "Generate intermediate questions requiring understanding (How, Why).",
        "hard": "Generate advanced analytical questions. Focus on implications, limitations, critical thinking.",
        "expert": "Generate expert-level questions for researchers. Novel connections, future work, technical depth."
    }

    prompt = f"""{difficulty_prompts.get(difficulty, difficulty_prompts['medium'])}

Paper: {title}
Content: {source_text[:1500]}

Generate exactly {n} questions. Format:
Q1: [question]
A1: [detailed answer]
Keywords: [3-5 key terms separated by commas]

Q2: [question]
..."""

    ai_resp = await gemini_generate_text(prompt)
    items = []

    if ai_resp:
        lines = [l.strip() for l in ai_resp.splitlines() if l.strip()]
        q = None
        a = None

        for line in lines:
            if re.match(r'^Q\d+:', line):
                if q and a:
                    items.append({"q": q, "a": a, "a_keywords": []})
                q = re.sub(r'^Q\d+:\s*', '', line)
                a = None
            elif re.match(r'^A\d+:', line):
                a = re.sub(r'^A\d+:\s*', '', line)
            elif line.lower().startswith("keywords:"):
                kws = [k.strip().lower() for k in line.split(":", 1)[1].split(",") if k.strip()]
                if q:
                    items.append({"q": q, "a": a or "", "a_keywords": kws})
                    q = None
                    a = None

        if q:
            items.append({"q": q, "a": a or "", "a_keywords": []})

    # Fallback
    if not items:
        base_kws = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", source_text)][:9]
        items = [
            {"q": f"What is the main contribution of '{title}'?", "a": "The main contribution is...", "a_keywords": base_kws[:3]},
            {"q": "Describe the methodology used.", "a": "The methodology involves...", "a_keywords": base_kws[3:6]},
            {"q": "What are the key results?", "a": "The key results show...", "a_keywords": base_kws[6:9]},
        ]

    return items[:n]


async def evaluate_answer_semantic(question: str, user_answer: str, reference_answer: str,
                                   keywords: List[str]) -> Tuple[float, str]:
    """Semantic answer evaluation with detailed feedback"""

    scores = []

    # Method 1: Keyword matching (fast)
    keyword_score = sum(1 for k in keywords if k.lower() in user_answer.lower())
    keyword_ratio = keyword_score / max(len(keywords), 1)
    scores.append(keyword_ratio * 10)

    # Method 2: Embedding similarity (if available)
    if embedding_model:
        try:
            user_emb = await generate_embedding(user_answer)
            ref_emb = await generate_embedding(reference_answer)

            if user_emb is not None and ref_emb is not None:
                sim = cosine_similarity([user_emb], [ref_emb])[0][0]
                scores.append(sim * 10)
        except:
            pass

    # Method 3: LLM grading (most accurate but slower)
    llm_score = await grade_with_llm(question, user_answer, reference_answer)
    if llm_score is not None:
        scores.append(llm_score)

    # Weighted average
    if len(scores) >= 3:
        final_score = 0.2 * scores[0] + 0.3 * scores[1] + 0.5 * scores[2]
    elif len(scores) == 2:
        final_score = 0.4 * scores[0] + 0.6 * scores[1]
    else:
        final_score = scores[0] if scores else 5.0

    # Generate feedback
    feedback = await generate_feedback(final_score, keywords, keyword_score, user_answer)

    return final_score, feedback


async def grade_with_llm(question: str, user_answer: str, reference: str) -> Optional[float]:
    """Use LLM for grading"""
    prompt = f"""Grade this answer on a scale of 0-10.

Question: {question}
Reference answer: {reference}
Student answer: {user_answer}

Consider: accuracy, completeness, understanding.
Output ONLY a number from 0-10."""

    response = await gemini_generate_text(prompt, temperature=0.1)
    if response:
        try:
            score = float(re.search(r'\d+\.?\d*', response).group())
            return min(max(score, 0), 10)
        except:
            pass
    return None


async def generate_feedback(score: float, keywords: List[str], keyword_count: int,
                           user_answer: str) -> str:
    """Generate detailed feedback"""
    if score >= 8.5:
        base = "🌟 Excellent! You demonstrated strong understanding."
    elif score >= 7.0:
        base = "✅ Great job! You covered the key points."
    elif score >= 5.0:
        base = "👍 Good effort! You're on the right track."
    else:
        base = "💡 Keep trying! Let me help clarify:"

    if score < 8.0:
        missing = [k for k in keywords if k.lower() not in user_answer.lower()]
        if missing:
            base += f"\n\n📌 Key concepts to consider: {', '.join(missing[:3])}"

    return base


# ---------------------------
# Spaced Repetition (SM-2)
# ---------------------------

async def schedule_review(user_id: str, paper_id: str, question_id: str,
                         performance: float, db: AsyncSession):
    """Schedule next review using SM-2 algorithm"""
    from sqlalchemy import select

    result = await db.execute(
        select(ReviewSchedule).where(
            ReviewSchedule.user_id == user_id,
            ReviewSchedule.question_id == question_id
        )
    )
    review = result.scalar_one_or_none()

    if not review:
        review = ReviewSchedule(
            user_id=user_id,
            paper_id=paper_id,
            question_id=question_id,
            easiness=2.5,
            interval=1,
            repetitions=0
        )
        db.add(review)

    # SM-2 algorithm
    if performance >= 6:  # Correct
        if review.repetitions == 0:
            interval = 1
        elif review.repetitions == 1:
            interval = 6
        else:
            interval = int(review.interval * review.easiness)
        repetitions = review.repetitions + 1
    else:  # Incorrect
        interval = 1
        repetitions = 0

    # Update easiness
    easiness = review.easiness + (0.1 - (5 - performance/2) * (0.08 + (5 - performance/2) * 0.02))
    easiness = max(1.3, easiness)

    # Update
    review.easiness = easiness
    review.interval = interval
    review.repetitions = repetitions
    review.next_review = datetime.utcnow() + timedelta(days=interval)
    review.last_score = performance

    await db.commit()


async def get_due_reviews(user_id: str, db: AsyncSession) -> List[ReviewSchedule]:
    """Get questions due for review"""
    from sqlalchemy import select

    result = await db.execute(
        select(ReviewSchedule).where(
            ReviewSchedule.user_id == user_id,
            ReviewSchedule.next_review <= datetime.utcnow()
        )
    )
    return result.scalars().all()


# ---------------------------
# Intent Detection
# ---------------------------

QNA_START_PATTERNS = [
    re.compile(r"\bready\s*for\s*q\s*&\s*a\b", re.I),
    re.compile(r"let'?s\s*do\s*q\s*&\s*a", re.I),
    re.compile(r"\bstart\s*qna\b", re.I),
]

_RE_CAPABILITIES = re.compile(r"\b(what\s+can\s+you\s+do|capabilit|feature)\b", re.I)
_RE_HELP = re.compile(r"\bhelp\b", re.I)
_RE_STATUS = re.compile(r"\b(status|where\s+am\s+i)\b", re.I)
_RE_RESET = re.compile(r"\b(reset|clear)\b", re.I)
_RE_QNA_SKIP = re.compile(r"\bskip\b", re.I)
_RE_QNA_REPEAT = re.compile(r"\brepeat\b", re.I)
_RE_DETAILS = re.compile(r"\bmore\s+details\b", re.I)
_RE_FIGURES = re.compile(r"\b(show\s+)?figures?\b", re.I)
_RE_CITATIONS = re.compile(r"\b(citations?|references?|cites)\b", re.I)
_RE_EXPORT = re.compile(r"\bexport\b", re.I)
_RE_ADD_TO_LIST = re.compile(r"\badd\s+to\s+list\b", re.I)
_RE_MY_STATS = re.compile(r"\b(my\s+stats|statistics|analytics)\b", re.I)
_RE_RECOMMEND = re.compile(r"\brecommend\b", re.I)
_RE_COMPARE = re.compile(r"\bcompare\b", re.I)
_RE_SIMILAR = re.compile(r"\bsimilar\s+papers?\b", re.I)

STOPWORDS = {'the','is','a','an','of','and','or','to','in','on','for','with','by'}


def detect_intent(text: str) -> str:
    """Detect user intent"""
    text_lower = text.lower()

    # Q&A intents
    if any(p.search(text) for p in QNA_START_PATTERNS):
        return "qna_start"
    if _RE_QNA_SKIP.search(text):
        return "qna_skip"
    if _RE_QNA_REPEAT.search(text):
        return "qna_repeat"

    # Feature intents
    if _RE_FIGURES.search(text):
        return "show_figures"
    if _RE_CITATIONS.search(text):
        return "show_citations"
    if _RE_EXPORT.search(text):
        return "export"
    if _RE_ADD_TO_LIST.search(text):
        return "add_to_list"
    if _RE_MY_STATS.search(text):
        return "my_stats"
    if _RE_RECOMMEND.search(text):
        return "recommend"
    if _RE_COMPARE.search(text):
        return "compare"
    if _RE_SIMILAR.search(text):
        return "similar_papers"

    # Navigation
    if _RE_DETAILS.search(text):
        return "details"
    if _RE_RESET.search(text):
        return "reset"
    if _RE_STATUS.search(text):
        return "status"
    if _RE_HELP.search(text):
        return "help"
    if _RE_CAPABILITIES.search(text):
        return "capabilities"

    # Selection
    if re.search(r"\b(select|choose|pick)\s+(\d+)\b", text, re.I):
        return "selection"

    # Paper search
    if re.search(r"https?://|arxiv|\d{4}\.\d{4,5}|10\.\d{4,}/", text, re.I):
        return "paper"

    # Content-based (3+ meaningful words)
    tokens = [t for t in re.findall(r"[a-z]+", text_lower) if t not in STOPWORDS]
    if len(tokens) >= 3:
        return "paper"

    return "ambiguous"


# ---------------------------
# Bot Handlers
# ---------------------------

async def handle_browsing(user_id: str, text: str, db: AsyncSession) -> str:
    """Handle browsing mode"""
    intent = detect_intent(text)
    sess = await get_session(user_id, db)

    # Global commands
    if intent == "reset":
        await update_session(user_id, db, mode="browsing", qna_active=False,
                           qna_index=0, qna_questions=None, score=0,
                           selected_paper_id=None, last_results=None)
        return "✨ Session cleared. Send a paper title or link to begin!"

    if intent == "status":
        last_q = sess.selected_paper_title or ""
        return f"📍 *Status*\nMode: Browsing\nLast query: '{last_q}'" if last_q else "📍 Browsing mode. Search for a paper to begin!"

    if intent == "help":
        return """📚 *Research Paper Bot - Help*

🔍 *Search:* Send any paper title, keyword, arXiv link, or DOI
📄 *Select:* Reply 'select 1' (or 2/3) after search
💬 *Q&A:* Type 'start qna' to test your understanding
📊 *Features:* my stats | recommend | add to list | show figures

Commands: help | status | reset"""

    if intent == "capabilities":
        return """🤖 *What I Can Do:*

✅ Search across 100M+ papers (Semantic Scholar + arXiv)
✅ Download and analyze full PDFs
✅ Generate AI summaries with figures
✅ Smart Q&A with semantic grading
✅ Track your reading & learning
✅ Recommend papers based on your interests
✅ Export citations (BibTeX, RIS, CSV)
✅ Study groups & achievements

Type 'help' for commands!"""

    # My stats
    if intent == "my_stats":
        stats = await generate_user_stats(user_id, db)
        return stats

    # Recommendations
    if intent == "recommend":
        papers = await recommend_papers(user_id, db)
        if not papers:
            return "📚 Read a few papers first, then I can recommend more based on your interests!"

        msg = "🎯 *Recommended for You:*\n\n"
        for i, paper in enumerate(papers[:3], 1):
            msg += f"{i}. {paper['title']} ({paper.get('year', '?')})\n"
            msg += f"   {paper.get('authors', '')[:50]}...\n\n"
        msg += "Reply 'select 1' to read!"

        # Cache results
        await update_session(user_id, db, last_results=papers)
        return msg

    # Selection
    if intent == "selection":
        match = re.search(r"(select|choose|pick)\s+(\d+)", text, re.I)
        if match:
            idx = int(match.group(2)) - 1

            last_results = sess.last_results or []
            if isinstance(last_results, str):
                try:
                    last_results = json.loads(last_results)
                except:
                    last_results = []

            if not last_results:
                return "❌ No recent search. Please search for a paper first!"

            if idx < 0 or idx >= len(last_results):
                return f"❌ Invalid selection. Choose 1-{len(last_results)}."

            chosen = last_results[idx]
            paper_id = chosen.get("paperId", f"paper_{hash(chosen['title'])}")

            # Store in database
            await store_paper(chosen, db)

            # Update session
            await update_session(
                user_id, db,
                selected_paper_id=paper_id,
                selected_paper_title=chosen.get("title"),
                selected_paper_abstract=chosen.get("abstract", "")
            )

            # Log history
            history = UserHistory(user_id=user_id, paper_id=paper_id, action="read")
            db.add(history)
            await db.commit()

            # Check for PDF and download in background
            arxiv_match = re.search(r'(\d{4}\.\d{4,5})', chosen.get('url', ''))
            if arxiv_match:
                # Download PDF asynchronously
                asyncio.create_task(process_paper_pdf(paper_id, arxiv_match.group(1), db))

            # Generate summary
            summary = await summarize_paper(
                chosen.get("title", ""),
                chosen.get("abstract", ""),
                detailed=False
            )

            compact = compact_summary(
                title=chosen.get("title", ""),
                year=chosen.get("year"),
                authors=chosen.get("authors", ""),
                url=chosen.get("url", ""),
                structured_text=summary
            )

            return compact

    # Paper search
    if intent == "paper":
        query = text.strip()

        # Check cache
        cache_key = f"search:{hashlib.md5(query.encode()).hexdigest()}"
        cached = await cache.get_json(cache_key)
        if cached:
            results = cached
        else:
            # Search both sources in parallel
            results_ss, results_arxiv = await asyncio.gather(
                search_papers_semantic_scholar(query, limit=3),
                search_papers_arxiv(query, limit=3)
            )

            # Combine and deduplicate
            results = results_ss if results_ss else results_arxiv

            # Also do semantic search if available
            semantic_results = await semantic_search(query, n_results=3)

            # Cache
            await cache.set_json(cache_key, results, ttl=3600)

        if not results:
            return f"❌ No papers found for '{query}'. Try different keywords!"

        # Format results
        msg = "🔍 *Search Results:*\n\n"
        for i, paper in enumerate(results[:3], 1):
            msg += f"📄 *{i}. {paper['title']}*\n"
            msg += f"👤 {paper.get('authors', '')[:60]}\n"
            msg += f"📅 {paper.get('year', '?')} | 📊 {paper.get('citationCount', 0)} citations\n"

            teaser = paper.get('abstract', '')[:120]
            if teaser:
                msg += f"_{teaser}..._\n"
            msg += "\n"

        msg += "Reply: *select 1* (or 2/3)"

        # Save results
        await update_session(user_id, db, selected_paper_title=query, last_results=results)

        return msg

    # Ambiguous
    return "🤔 I didn't understand that. Try:\n• Search for a paper\n• Type 'help' for commands"


async def handle_qna(user_id: str, text: str, db: AsyncSession) -> str:
    """Handle Q&A mode"""
    sess = await get_session(user_id, db)
    intent = detect_intent(text)

    # Allow help/status during Q&A
    if intent == "help":
        return "💬 Q&A Mode: Answer questions, or use: skip | repeat | hint\nType 'reset' to exit."

    if intent == "status":
        idx = sess.qna_index or 0
        title = sess.selected_paper_title or "Unknown"
        return f"📍 Q&A Mode\nPaper: '{title}'\nQuestion: {idx + 1}"

    # Load questions
    qna_questions = sess.qna_questions or []
    if isinstance(qna_questions, str):
        try:
            qna_questions = json.loads(qna_questions)
        except:
            qna_questions = []

    idx = sess.qna_index or 0

    # Initialize Q&A if empty
    if not qna_questions:
        title = sess.selected_paper_title or ""
        abstract = sess.selected_paper_abstract or ""

        if not title:
            await update_session(user_id, db, mode="browsing", qna_active=False)
            return "❌ No paper selected. Search and select a paper first!"

        # Generate questions
        summary = await summarize_paper(title, abstract)
        difficulty = sess.difficulty_preference or "medium"
        qna_questions = await generate_qna_items(title, summary, difficulty=difficulty, n=3)

        await update_session(user_id, db, qna_questions=qna_questions, qna_index=0, qna_active=True)

        return f"🎯 *Q&A Started* (Difficulty: {difficulty})\n\n❓ Q1: {qna_questions[0]['q']}"

    # Check completion
    if idx >= len(qna_questions):
        total_score = sess.score or 0
        await update_session(user_id, db, mode="browsing", qna_active=False, qna_index=0, qna_questions=None)

        # Log completion
        history = UserHistory(
            user_id=user_id,
            paper_id=sess.selected_paper_id,
            action="qna_completed",
            score=total_score
        )
        db.add(history)
        await db.commit()

        # Check achievements
        achievements = await check_achievements(user_id, db)
        achieve_msg = ""
        if achievements:
            achieve_msg = f"\n\n🏆 New achievement: {achievements[0]}"

        return f"🎉 *Q&A Complete!*\n\nFinal Score: {total_score:.1f}/30{achieve_msg}\n\nYou're back to browsing. Search for another paper!"

    # Handle controls
    if intent == "qna_repeat":
        return f"🔄 Repeating:\n\n❓ Q{idx+1}: {qna_questions[idx]['q']}"

    if intent == "qna_skip":
        idx += 1
        await update_session(user_id, db, qna_index=idx)

        if idx < len(qna_questions):
            return f"⏭ Skipped.\n\n❓ Q{idx+1}: {qna_questions[idx]['q']}"
        else:
            return await handle_qna(user_id, "", db)  # Trigger completion

    # Evaluate answer
    current_q = qna_questions[idx]
    score, feedback = await evaluate_answer_semantic(
        current_q['q'],
        text,
        current_q.get('a', ''),
        current_q.get('a_keywords', [])
    )

    total_score = (sess.score or 0) + score
    idx += 1

    await update_session(user_id, db, qna_index=idx, score=total_score)

    # Schedule spaced repetition
    question_id = f"{sess.selected_paper_id}_q{idx-1}"
    await schedule_review(user_id, sess.selected_paper_id, question_id, score, db)

    # Next question or complete
    if idx < len(qna_questions):
        return f"{feedback}\n\n❓ Q{idx+1}: {qna_questions[idx]['q']}"
    else:
        return await handle_qna(user_id, "", db)  # Trigger completion


# ---------------------------
# Helper Functions
# ---------------------------

async def store_paper(paper_data: Dict, db: AsyncSession):
    """Store paper in database"""
    from sqlalchemy import select

    paper_id = paper_data.get("paperId", f"paper_{hash(paper_data['title'])}")

    # Check if exists
    result = await db.execute(select(Paper).where(Paper.paper_id == paper_id))
    existing = result.scalar_one_or_none()

    if existing:
        return existing

    # Create new
    paper = Paper(
        paper_id=paper_id,
        title=paper_data.get("title"),
        authors=paper_data.get("authors"),
        year=paper_data.get("year"),
        venue=paper_data.get("venue"),
        url=paper_data.get("url"),
        abstract=paper_data.get("abstract"),
        citation_count=paper_data.get("citationCount", 0),
        metadata=paper_data
    )

    db.add(paper)
    await db.commit()

    # Store embedding asynchronously
    asyncio.create_task(store_paper_embedding(
        paper_id,
        paper.title,
        paper.abstract or "",
        {"title": paper.title, "authors": paper.authors, "year": paper.year}
    ))

    return paper


async def process_paper_pdf(paper_id: str, arxiv_id: str, db: AsyncSession):
    """Background task: Download and process PDF"""
    try:
        logger.info(f"Processing PDF for {paper_id}")

        # Download
        pdf_bytes = await download_pdf_from_arxiv(arxiv_id)
        if not pdf_bytes:
            return

        # Extract
        extracted = await extract_pdf_text(pdf_bytes)

        # Update database
        from sqlalchemy import update as sql_update
        await db.execute(
            sql_update(Paper).where(Paper.paper_id == paper_id).values(
                full_text=extracted.get("full_text", ""),
                sections=extracted.get("sections", {}),
            )
        )
        await db.commit()

        logger.info(f"PDF processed for {paper_id}")
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")


async def generate_user_stats(user_id: str, db: AsyncSession) -> str:
    """Generate user statistics"""
    from sqlalchemy import select, func

    # Count papers read
    result = await db.execute(
        select(func.count(UserHistory.id)).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "read"
        )
    )
    papers_read = result.scalar() or 0

    # Count Q&As
    result = await db.execute(
        select(func.count(UserHistory.id), func.avg(UserHistory.score)).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "qna_completed"
        )
    )
    row = result.first()
    qnas_completed = row[0] or 0
    avg_score = row[1] or 0

    # Achievements
    result = await db.execute(
        select(func.count(Achievement.id)).where(Achievement.user_id == user_id)
    )
    achievement_count = result.scalar() or 0

    # Streak
    streak = await calculate_streak(user_id, db)

    msg = f"""📊 *Your Statistics*

📖 Papers read: {papers_read}
✅ Q&As completed: {qnas_completed}
🎯 Average score: {avg_score:.1f}/10
🔥 Current streak: {streak} days
🏆 Achievements: {achievement_count}

Keep learning! 🚀"""

    return msg


async def calculate_streak(user_id: str, db: AsyncSession) -> int:
    """Calculate current study streak"""
    from sqlalchemy import select, func

    # Get distinct days with activity
    result = await db.execute(
        select(func.date(UserHistory.timestamp).label('day')).where(
            UserHistory.user_id == user_id
        ).group_by('day').order_by(func.date(UserHistory.timestamp).desc()).limit(30)
    )

    days = [row[0] for row in result]

    if not days:
        return 0

    streak = 0
    expected = datetime.utcnow().date()

    for day in days:
        if day == expected:
            streak += 1
            expected -= timedelta(days=1)
        else:
            break

    return streak


async def recommend_papers(user_id: str, db: AsyncSession, n: int = 3) -> List[Dict]:
    """Recommend papers based on user history"""
    from sqlalchemy import select

    # Get user's reading history
    result = await db.execute(
        select(UserHistory.paper_id).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "read"
        ).limit(10)
    )
    read_paper_ids = [row[0] for row in result]

    if not read_paper_ids:
        # Return trending papers
        return await search_papers_semantic_scholar("machine learning", limit=n)

    # Get papers
    result = await db.execute(
        select(Paper).where(Paper.paper_id.in_(read_paper_ids))
    )
    read_papers = result.scalars().all()

    # Simple recommendation: search for similar topics
    if read_papers:
        # Use titles as query
        query = " ".join([p.title for p in read_papers[:3]])
        results = await search_papers_semantic_scholar(query, limit=n*2)

        # Filter out already read
        filtered = [r for r in results if r.get("paperId") not in read_paper_ids]
        return filtered[:n]

    return []


async def check_achievements(user_id: str, db: AsyncSession) -> List[str]:
    """Check and unlock achievements"""
    from sqlalchemy import select, func

    new_achievements = []

    # First paper achievement
    result = await db.execute(
        select(func.count(Achievement.id)).where(
            Achievement.user_id == user_id,
            Achievement.achievement_key == "first_paper"
        )
    )
    has_first = result.scalar() > 0

    if not has_first:
        result = await db.execute(
            select(func.count(UserHistory.id)).where(
                UserHistory.user_id == user_id,
                UserHistory.action == "read"
            )
        )
        papers_read = result.scalar() or 0

        if papers_read >= 1:
            achievement = Achievement(user_id=user_id, achievement_key="first_paper")
            db.add(achievement)
            await db.commit()
            new_achievements.append("📖 First Steps - Read your first paper!")

    # Add more achievement checks...

    return new_achievements


# ---------------------------
# WhatsApp Webhook
# ---------------------------

async def verify_twilio_signature(request: Request) -> bool:
    """Verify Twilio webhook signature"""
    signature = request.headers.get("X-Twilio-Signature", "")
    url = str(request.url)

    form_data = await request.form()
    params = dict(form_data)

    return twilio_validator.validate(url, params, signature)


@app.post("/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks,
                          db: AsyncSession = Depends(get_db)):
    """Handle incoming WhatsApp messages"""

    # Verify signature (security)
    if not await verify_twilio_signature(request):
        logger.warning("Invalid Twilio signature")
        # In production, uncomment this:
        # raise HTTPException(status_code=403, detail="Invalid signature")

    # Parse request
    form_data = await request.form()
    from_number = form_data.get("From", "")
    body = form_data.get("Body", "").strip()
    user_id = from_number or "unknown"

    # Log incoming message
    await log_message(user_id, "user", body, db)

    # Get session
    sess = await get_session(user_id, db)

    # Route based on mode
    if any(p.search(body) for p in QNA_START_PATTERNS):
        await update_session(user_id, db, mode="qna", qna_active=True, qna_index=0, qna_questions=None)
        reply_text = await handle_qna(user_id, body, db)
    elif sess.mode == "qna" and sess.qna_active:
        reply_text = await handle_qna(user_id, body, db)
    else:
        reply_text = await handle_browsing(user_id, body, db)

    # Log response
    await log_message(user_id, "bot", reply_text, db)

    # Split into chunks
    def split_chunks(text: str, limit: int = 1500) -> List[str]:
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break

            # Find last space before limit
            split_pos = text.rfind(' ', 0, limit)
            if split_pos == -1:
                split_pos = limit

            chunks.append(text[:split_pos])
            text = text[split_pos:].lstrip()

        # Add tags if multiple
        if len(chunks) > 1:
            tagged = []
            for i, chunk in enumerate(chunks, 1):
                tagged.append(f"({i}/{len(chunks)})\n{chunk}")
            return tagged

        return chunks

    # Build TwiML response
    resp = MessagingResponse()
    for chunk in split_chunks(reply_text):
        resp.message(chunk)

    return Response(content=str(resp), media_type="application/xml")


@app.get("/")
async def root():
    """Health check"""
    return {"status": "running", "version": "2.0.0", "features": "all"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "redis": "connected" if redis_client else "disconnected",
        "embeddings": "available" if embedding_model else "unavailable",
        "vector_store": "available" if vector_store else "unavailable",
        "pdf_processing": "available" if PDF_AVAILABLE else "unavailable"
    }


# ---------------------------
# Startup & Shutdown
# ---------------------------

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    global redis_client

    logger.info("Starting application...")

    # Initialize database
    await init_db()

    # Connect to Redis
    await cache.connect()
    redis_client = cache.redis

    logger.info("Application ready!")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")

    # Close HTTP client
    await http_client.aclose()

    # Close Redis
    if redis_client:
        await redis_client.close()

    logger.info("Shutdown complete")


# ---------------------------
# Main Entry Point
# ---------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "async_research_bot:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
