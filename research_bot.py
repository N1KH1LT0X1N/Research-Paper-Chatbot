import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import logging
import hashlib
import time
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from cachetools import TTLCache


# ---------------------------
# Config & Initialization
# ---------------------------

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "whatsapp_bot.db")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "30"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g., whatsapp:+14155238886
TWILIO_VALIDATE_WEBHOOK = os.getenv("TWILIO_VALIDATE_WEBHOOK", "true").lower() == "true"
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL", "")

if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    missing = []
    if not TWILIO_ACCOUNT_SID:
        missing.append("TWILIO_ACCOUNT_SID")
    if not TWILIO_AUTH_TOKEN:
        missing.append("TWILIO_AUTH_TOKEN")
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}. Set them in .env."
    )

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)

# ---------------------------
# Logging Configuration
# ---------------------------
logger = logging.getLogger("research_bot")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(_handler)

# ---------------------------
# Response Caching (TTL: 1 hour)
# ---------------------------
_summary_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)
_qna_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)
_section_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)

def _cache_key(*args: str) -> str:
    """Generate a stable cache key from arguments."""
    return hashlib.sha256(":".join(str(a) for a in args).encode()).hexdigest()[:16]

# ---------------------------
# Twilio Webhook Validation
# ---------------------------
def validate_twilio_request(f):
    """Decorator to validate Twilio webhook signatures."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not TWILIO_VALIDATE_WEBHOOK:
            return f(*args, **kwargs)
        
        if not TWILIO_AUTH_TOKEN:
            logger.error("TWILIO_AUTH_TOKEN not set, cannot validate webhook")
            return "Unauthorized", 403
        
        validator = RequestValidator(TWILIO_AUTH_TOKEN)
        url = TWILIO_WEBHOOK_URL or request.url
        signature = request.headers.get("X-Twilio-Signature", "")
        
        if not validator.validate(url, request.form.to_dict(), signature):
            logger.warning(f"Invalid Twilio signature from {request.remote_addr}")
            return "Forbidden", 403
        
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------
# Database (SQLite) helpers
# ---------------------------

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            user_id TEXT PRIMARY KEY,
            mode TEXT DEFAULT 'browsing',   -- 'browsing' | 'qna'
            selected_paper_id TEXT,
            selected_paper_title TEXT,
            selected_paper_abstract TEXT,
            qna_active INTEGER DEFAULT 0,
            qna_index INTEGER DEFAULT 0,
            qna_questions TEXT,             -- JSON list of {q, a_keywords}
            score INTEGER DEFAULT 0,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,      -- 'user' | 'bot'
            message TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    # Lightweight schema migrations for new columns
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN last_results TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN prompt_version TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN last_ambiguous TEXT")
    except sqlite3.OperationalError:
        pass
    conn.close()


def log_message(user_id: str, role: str, message: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO logs (user_id, role, message, created_at) VALUES (?, ?, ?, ?)",
        (user_id, role, message, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


def get_session(user_id: str) -> sqlite3.Row:
    conn = get_db_connection()
    cur = conn.execute("SELECT * FROM sessions WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO sessions (user_id, updated_at) VALUES (?, ?)",
            (user_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        cur = conn.execute("SELECT * FROM sessions WHERE user_id=?", (user_id,))
        row = cur.fetchone()
    conn.close()
    assert row is not None
    return row


def update_session(user_id: str, **kwargs: Any) -> None:
    if not kwargs:
        return
    columns = ", ".join([f"{k}=?" for k in kwargs.keys()])
    values = list(kwargs.values())
    values.append(user_id)
    conn = get_db_connection()
    conn.execute(
        f"UPDATE sessions SET {columns}, updated_at=? WHERE user_id=?",
        [*kwargs.values(), datetime.now(timezone.utc).isoformat(), user_id],
    )
    conn.commit()
    conn.close()


# ---------------------------
# External APIs: Paper search
# ---------------------------

def http_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 8) -> Optional[requests.Response]:
    """GET with one retry and small backoff."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception:
            return None

def search_papers_semantic_scholar(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Search papers using Semantic Scholar's Graph API. Falls back to stub on error."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,url,abstract",
        "limit": limit,
    }
    r = http_get(url, params=params, timeout=8)
    if r is not None:
        try:
            data = r.json()
            results = []
            for item in data.get("data", [])[:limit]:
                results.append(
                    {
                        "paperId": item.get("paperId"),
                        "title": item.get("title"),
                        "year": item.get("year"),
                        "url": item.get("url"),
                        "authors": ", ".join([a.get("name", "") for a in item.get("authors", []) if a]),
                        "abstract": item.get("abstract") or "",
                    }
                )
            if results:
                return results
        except Exception:
            pass
    # Fallback stub
    return [
        {
            "paperId": "stub1",
            "title": "Attention Is All You Need",
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762",
            "authors": "Vaswani et al.",
            "abstract": (
                "We propose the Transformer, a new architecture that eschews recurrence and convolutions, "
                "relying entirely on attention mechanisms for sequence modeling."
            ),
        },
        {
            "paperId": "stub2",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2018,
            "url": "https://arxiv.org/abs/1810.04805",
            "authors": "Devlin et al.",
            "abstract": "We introduce BERT, a deeply bidirectional language representation model...",
        },
        {
            "paperId": "stub3",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "year": 2020,
            "url": "https://arxiv.org/abs/2005.14165",
            "authors": "Brown et al.",
            "abstract": "We present GPT-3, a 175B parameter autoregressive language model...",
        },
    ]


def search_papers_arxiv(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Very lightweight arXiv API fallback; minimal XML parsing to avoid extra deps."""
    base = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    r = http_get(base, params=params, timeout=8)
    if r is None:
        return []
    text = r.text
    entries = text.split("<entry>")[1:limit+1]
    results: List[Dict[str, Any]] = []
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
        if title:
            results.append({
                "paperId": None,
                "title": title,
                "year": year,
                "url": link,
                "authors": authors,
                "abstract": summary,
            })
    return results[:limit]


def extract_query_from_text(text: str) -> str:
    """Extract a search query from user text. If a URL/DOI/arXiv link exists, prefer a focused query.
    Falls back to the raw text if nothing is parsed.
    """
    text = text.strip()
    # DOI link
    m = re.search(r"https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.I)
    if m:
        return m.group(1)
    # arXiv link
    m = re.search(r"https?://arxiv\.org/(abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", text, re.I)
    if m:
        return m.group(2)
    # Any URL - use as query string
    m = re.search(r"https?://\S+", text)
    if m:
        return m.group(0)
    return text


# ---------------------------
# AI helpers (Groq LLM)
# ---------------------------

def groq_generate_text(prompt: str, temperature: float = TEMPERATURE) -> Optional[str]:
    """Generate text using Groq API with retry logic and logging."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not configured, LLM features disabled")
        return None
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    
    for attempt in range(2):  # 1 retry
        start_time = time.time()
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=GROQ_TIMEOUT,
            )
            latency_ms = (time.time() - start_time) * 1000
            
            if resp.status_code == 429:
                logger.warning(f"Groq rate limit hit, attempt {attempt + 1}")
                if attempt == 0:
                    time.sleep(2)  # Brief backoff before retry
                    continue
                return None
            
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Log success
            usage = data.get("usage", {})
            logger.info(
                f"Groq call: {latency_ms:.0f}ms, "
                f"tokens={usage.get('total_tokens', 'N/A')}, "
                f"model={GROQ_MODEL}"
            )
            
            return content.strip() if content else None
            
        except requests.exceptions.Timeout:
            logger.warning(f"Groq timeout on attempt {attempt + 1}")
            if attempt == 0:
                continue
            return None
        except requests.exceptions.RequestException as e:
            latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Groq API error: {type(e).__name__}: {e}, latency={latency_ms:.0f}ms")
            if attempt == 0:
                continue
            return None
    return None


def summarize_paper(title: str, abstract: str, url: str, detailed: bool = False) -> str:
    """Return a structured summary with sections: Introduction, Methodology, Results, Conclusions.
    If detailed=True, allow a bit more depth. Uses caching.
    """
    # Check cache first
    cache_key = _cache_key(title, abstract, str(detailed))
    if cache_key in _summary_cache:
        logger.debug(f"Cache hit for summary: {cache_key}")
        return _summary_cache[cache_key]
    
    # Llama-optimized prompt with explicit format
    sentence_count = "4-6 sentences" if detailed else "2-4 sentences"
    prompt = f"""You are a research paper summarizer. Summarize the paper below.

TASK: Create a structured summary with exactly 4 sections.
FORMAT: Use these exact markdown headers:
## Introduction
## Methodology
## Results
## Conclusions

REQUIREMENTS:
- Each section: {sentence_count}
- Be factual and precise
- If information is not in the abstract, write "Not specified in abstract."

PAPER:
Title: {title}
Abstract: {abstract}

OUTPUT (respond only with the formatted summary):"""

    ai_resp = groq_generate_text(prompt)
    if ai_resp:
        _summary_cache[cache_key] = ai_resp
        return ai_resp
    # Fallback structured summary if AI not configured
    def take(sn: str) -> str:
        return (sn[:400] + ("..." if len(sn) > 400 else "")) if sn else ""
    intro = take(abstract)
    method = "High-level method not available from abstract."
    results = "Key results are not explicitly stated in the abstract."
    concl = "Conclusions are inferred from context and may be limited."
    return (
        f"## Introduction\n{intro}\n\n"
        f"## Methodology\n{method}\n\n"
        f"## Results\n{results}\n\n"
        f"## Conclusions\n{concl}"
    )


def _parse_structured_sections(text: str) -> Dict[str, str]:
    """Parse the four sections from a markdown-like structured summary."""
    sections = {"Introduction": "", "Methodology": "", "Results": "", "Conclusions": ""}
    current = None
    lines = text.splitlines()
    for line in lines:
        m = re.match(r"^##\s*(Introduction|Methodology|Results|Conclusions)\s*$", line.strip(), re.I)
        if m:
            name = m.group(1).capitalize()
            # Normalize plural/singular
            if name.lower().startswith("conclusion"):
                name = "Conclusions"
            current = name
            continue
        if current:
            sections[current] += (line + "\n")
    # Trim whitespace
    for k in sections:
        sections[k] = sections[k].strip()
    return sections


def _sentences(text: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    # Clean empties and overly long whitespace
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]


def _join_sents(sents: List[str], max_sents: int) -> str:
    return " ".join(sents[:max_sents]).strip()


def _format_whatsapp_summary(title: str, year: Optional[int], authors: str, url: str, sections: Dict[str, str]) -> str:
    # Shorten authors if too long
    auth = re.sub(r"\s+", " ", authors or "").strip()
    if len(auth) > 80:
        # keep first 2 names if possible
        parts = [p.strip() for p in auth.split(",") if p.strip()]
        if len(parts) >= 2:
            auth = f"{parts[0]}, {parts[1]} et al."
        else:
            auth = auth[:77] + "..."
    title_line = f"{title} ({year}) by {auth}" if year else f"{title} by {auth}"
    header = title_line.strip() + (f"\n{url.strip()}" if url else "")

    # Bold headings for WhatsApp
    body_parts = []
    for key in ["Introduction", "Methodology", "Results", "Conclusions"]:
        content = re.sub(r"\s+", " ", sections.get(key, "")).strip()
        body_parts.append(f"\n\n*{key}*\n{content}" if content else f"\n\n*{key}*\n")
    return header + "".join(body_parts)


def compact_summary_to_single_message(title: str, year: Optional[int], authors: str, url: str, structured_text: str, limit: int = 1400) -> str:
    """Return a single-message WhatsApp-friendly summary under the given char limit, with CTA."""
    sections = _parse_structured_sections(structured_text)
    # Try progressive compaction: 4->2->1 sentences per section
    for max_sents in [4, 3, 2, 1]:
        compacted: Dict[str, str] = {}
        for k, v in sections.items():
            sents = _sentences(v)
            compacted[k] = _join_sents(sents, max_sents)
        msg = _format_whatsapp_summary(title, year, authors, url, compacted)
        cta = "\n\nWhat's next: start qna | more details intro|method|results|conclusions"
        full = f"{msg}{cta}"
        if len(full) <= limit:
            return full
    # As last resort, hard truncate while preserving structure
    msg = _format_whatsapp_summary(title, year, authors, url, sections)
    cta = "\n\nWhat's next: start qna | more details intro|method|results|conclusions"
    full = f"{msg}{cta}"
    if len(full) > limit:
        return full[:limit-3] + "..."
    return full


def generate_qna_items(title: str, source_text: str, n: int = 3) -> List[Dict[str, Any]]:
    """Generate Q&A based on the structured summary (preferred) or abstract/title fallback. Uses caching."""
    # Check cache first
    cache_key = _cache_key(title, source_text, str(n))
    if cache_key in _qna_cache:
        logger.debug(f"Cache hit for Q&A: {cache_key}")
        return _qna_cache[cache_key]
    
    # Llama-optimized prompt with few-shot example
    prompt = f"""You are generating quiz questions about a research paper.

TASK: Create exactly {n} questions with answer keywords.

FORMAT (use this exact format for each):
1. [Question text]
Answer keywords: keyword1, keyword2, keyword3

2. [Question text]
Answer keywords: keyword1, keyword2, keyword3

EXAMPLE:
1. What architecture does the paper propose?
Answer keywords: transformer, attention, encoder-decoder

REQUIREMENTS:
- Questions should test understanding of key concepts
- Keywords should be 3-5 single words from the paper
- Make questions specific and answerable

PAPER: {title}

SUMMARY:
{source_text}

QUESTIONS:"""

    ai_resp = groq_generate_text(prompt)
    items: List[Dict[str, Any]] = []
    if ai_resp:
        # Robust parsing for Llama output variations
        lines = [l.strip() for l in ai_resp.splitlines() if l.strip()]
        q: Optional[str] = None
        for line in lines:
            # Match variations: "1.", "1)", "Q1:", "Question 1:"
            q_match = re.match(r'^(?:\d+[.\):]|Q\d+:?|Question\s+\d+:?)\s*(.+)', line, re.I)
            # Match variations: "Answer keywords:", "Keywords:", "Key words:"
            kw_match = re.match(r'^(?:answer\s+)?key\s*words?:?\s*(.+)', line, re.I)
            
            if q_match:
                if q:
                    items.append({"q": q, "a_keywords": []})
                q = q_match.group(1).strip()
            elif kw_match and q:
                kws = [k.strip().lower() for k in kw_match.group(1).split(",") if k.strip()]
                items.append({"q": q, "a_keywords": kws})
                q = None
        if q:
            items.append({"q": q, "a_keywords": []})
    
    if items:
        _qna_cache[cache_key] = items[:n]
        return items[:n]
    if not items:
        # Fallback deterministic items from abstract/title
        base_kws = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", source_text)][:8]
        items = [
            {"q": f"What is the main idea of '{title}'?", "a_keywords": base_kws[:3]},
            {"q": "Name one key contribution.", "a_keywords": base_kws[3:6]},
            {"q": "Mention a limitation or challenge.", "a_keywords": base_kws[6:8]},
        ]
    return items[:n]


def evaluate_answer(user_text: str, a_keywords: List[str]) -> Tuple[int, str]:
    if not a_keywords:
        return 1, "Thanks!"
    text = user_text.lower()
    score = sum(1 for k in a_keywords if k and k in text)
    if score >= max(1, len(a_keywords) // 2):
        return score, "Great! You covered key points."
    hint = ", ".join(a_keywords[:2]) if a_keywords else ""
    return score, f"Good try! Consider: {hint}" if hint else "Good try!"


# ---------------------------
# Bot logic
# ---------------------------

QNA_START_PATTERNS = [
    re.compile(r"\bready\s*for\s*(?:q\s*[&a]?\s*a|qna)\b", re.I),
    re.compile(r"(?:let'?s\s*)?(?:do\s+)?(?:q\s*[&a]?\s*a|qna)", re.I),
    re.compile(r"\b(?:start\s+)?(?:q\s*[&a]?\s*a|qna)\b", re.I),
]


def is_qna_start(text: str) -> bool:
    return any(p.search(text) for p in QNA_START_PATTERNS)


# ---------------------------
# Intent routing (rules-based)
# ---------------------------

_RE_CAPABILITIES = re.compile(r"\b(what\s+can\s+you\s+do|what\s+do\s+you\s+do|capabilit|feature|how\s+to\s+use)\b", re.I)
_RE_HELP = re.compile(r"\bhelp\b", re.I)
_RE_STATUS = re.compile(r"\b(status|where\s+am\s+i)\b", re.I)
_RE_RESET = re.compile(r"\b(reset|end\s+session|clear)\b", re.I)
_RE_QNA_SKIP = re.compile(r"\bskip\b", re.I)
_RE_QNA_REPEAT = re.compile(r"\brepeat\b", re.I)
_RE_DETAILS = re.compile(r"\bmore\s+details\b", re.I)
_RE_URL = re.compile(r"https?://\S+", re.I)
_RE_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
_RE_ARXIV = re.compile(r"\b\d{4}\.\d{4,5}(v\d+)?\b")
_RE_PAPER_KEYWORDS = re.compile(r"\b(paper|research|study|arxiv|doi|preprint)\b", re.I)
_RE_SEARCH_CONFIRM = re.compile(r"\bsearch\b", re.I)


STOPWORDS = {
    'the','is','a','an','of','and','or','to','in','on','for','with','by','it','this','that','what','can','you','do','how','use','i','me','my','we','our'
}


def _token_count(text: str) -> int:
    toks = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text)]
    content = [t for t in toks if t not in STOPWORDS]
    return len(content)


def detect_intent(text: str) -> str:
    """Return one of: capabilities|help|status|reset|qna_start|qna_skip|qna_repeat|details|search_confirm|paper|ambiguous."""
    if is_qna_start(text):
        return "qna_start"
    if _RE_QNA_SKIP.search(text):
        return "qna_skip"
    if _RE_QNA_REPEAT.search(text):
        return "qna_repeat"
    if _RE_DETAILS.search(text):
        return "details"
    if _RE_SEARCH_CONFIRM.search(text):
        return "search_confirm"
    if _RE_RESET.search(text):
        return "reset"
    if _RE_STATUS.search(text):
        return "status"
    if _RE_HELP.search(text):
        return "help"
    if _RE_CAPABILITIES.search(text):
        return "capabilities"
    # paper indicators
    if _RE_URL.search(text) or _RE_DOI.search(text) or _RE_ARXIV.search(text) or _RE_PAPER_KEYWORDS.search(text):
        return "paper"
    # topic-like if enough content tokens
    if _token_count(text) >= 3:
        return "paper"
    return "ambiguous"


def capabilities_message() -> str:
    return (
        "I help you find and learn research papers. Send a title or link, select a result, get a structured "
        "summary, and start Q&A.\n\nWhat's next: start qna | help | reset"
    )


def help_message() -> str:
    return (
        "Commands: send a paper title or link (DOI/arXiv). select 1 to choose. start qna to quiz. "
        "more details intro|method|results|conclusions. skip/repeat during Q&A. reset to start over. status to see where you are."
    )


def status_message(sess: sqlite3.Row) -> str:
    mode = (sess["mode"] or "browsing").lower()
    if mode == "qna":
        idx = int(sess["qna_index"] or 0)
        title = sess["selected_paper_title"] or ""
        return f"Q&A on '{title}' (Q{idx+1}). Reply with your answer, or say skip/repeat."
    last_q = sess["selected_paper_title"] or ""
    return f"Browsing. Last query: '{last_q}'. No paper selected." if last_q else "Browsing. Send a paper title or link to begin."


def format_search_results(results: List[Dict[str, Any]]) -> str:
    lines = ["Here are the top results:"]
    for i, r in enumerate(results, 1):
        teaser = (r.get('abstract') or '').strip()
        if teaser:
            teaser = re.sub(r"\s+", " ", teaser)
            if len(teaser) > 120:
                teaser = teaser[:117] + "..."
        lines.append(f"{i}. {r['title']} ({r.get('year','?')}) - {r.get('authors','')}\n{r.get('url','')}\n{teaser}")
    lines.append("\nReply 'select 1' (or 2/3) to choose a paper.")
    return "\n".join(lines)


def handle_browsing(user_id: str, text: str) -> str:
    intent = detect_intent(text)

    # Session/global intents
    if intent == "reset":
        update_session(
            user_id,
            mode="browsing",
            qna_active=0,
            qna_index=0,
            qna_questions=json.dumps([]),
            score=0,
            selected_paper_id=None,
            selected_paper_title=None,
            selected_paper_abstract=None,
            last_results=json.dumps([]),
            last_ambiguous=None,
        )
        return "Session cleared. Send a paper title or link to begin."
    if intent == "status":
        return status_message(get_session(user_id))
    if intent == "help":
        return help_message()
    if intent == "capabilities":
        return capabilities_message()

    # Case-insensitive selection, also accept choose/pick/option and small spelled numbers
    sel_match = re.search(r"\b(select|choose|pick|option)\s+(one|two|three|\d+)\b", text, re.I)
    if sel_match:
        num_token = sel_match.group(2).lower()
        word_to_num = {"one": 1, "two": 2, "three": 3}
        try:
            idx_num = int(num_token) if num_token.isdigit() else word_to_num.get(num_token, 0)
        except Exception:
            idx_num = 0
        if idx_num <= 0:
            return "Please reply like 'select 1'."
        idx = idx_num - 1
        sess = get_session(user_id)
        last_results = []
        if sess["last_results"]:
            try:
                last_results = json.loads(sess["last_results"]) or []
            except Exception:
                last_results = []
        papers = last_results
        if not papers:
            return "No recent search to select from. Please search again."
        if idx < 0 or idx >= len(papers):
            return "Invalid selection. Please reply with 'select 1' (or 2/3)."
        chosen = papers[idx]
        update_session(
            user_id,
            selected_paper_id=chosen.get("paperId"),
            selected_paper_title=chosen.get("title"),
            selected_paper_abstract=chosen.get("abstract", ""),
        )
        raw_summary = summarize_paper(chosen.get("title", ""), chosen.get("abstract", ""), chosen.get("url", ""))
        # Build a single-message, compact WhatsApp-friendly summary with CTA; do NOT auto-start Q&A
        compact = compact_summary_to_single_message(
            title=chosen.get("title", ""),
            year=chosen.get("year"),
            authors=chosen.get("authors", ""),
            url=chosen.get("url", ""),
            structured_text=raw_summary,
        )
        return compact

    # Clarify if ambiguous
    if intent == "ambiguous":
        update_session(user_id, last_ambiguous=text.strip())
        return (
            f"Do you want me to search for papers on '{text.strip()}', or see what I can do? "
            "Reply: search | help"
        )

    # If user confirms 'search', prefer last ambiguous text
    if intent == "search_confirm":
        sess = get_session(user_id)
        query_source = (sess["last_ambiguous"] or text).strip()
        if not query_source or len(query_source) < 3:
            return "Please provide a longer search query (at least 3 characters)."
        query = extract_query_from_text(query_source)
        results = search_papers_semantic_scholar(query)
        if not results:
            results = search_papers_arxiv(query)
        if not results:
            return "I couldn't find any papers. Try a different query."
        try:
            update_session(user_id, selected_paper_title=query, last_results=json.dumps(results), last_ambiguous=None)
        except Exception:
            update_session(user_id, selected_paper_title=query, last_ambiguous=None)
        return format_search_results(results)

    # Otherwise, treat as search query if intent is paper
    if intent != "paper":
        return "I didn't catch that. Send a paper title or link, or type help."
    # Also handle direct links (DOI / arXiv / general URL)
    query = extract_query_from_text(text)
    if len(query) < 3:
        return "Please provide a longer search query (at least 3 characters)."
    results = search_papers_semantic_scholar(query)
    if not results:
        # Try arXiv fallback
        results = search_papers_arxiv(query)
    if not results:
        return "I couldn't find any papers. Try a different query."
    # Save the query in session (so a naive 'select' can work)
    try:
        update_session(user_id, selected_paper_title=query, last_results=json.dumps(results), last_ambiguous=None)
    except Exception:
        update_session(user_id, selected_paper_title=query, last_ambiguous=None)
    return format_search_results(results)


def handle_qna(user_id: str, text: str) -> str:
    sess = get_session(user_id)
    intent = detect_intent(text)
    # Allow capabilities/help/status within Q&A without losing context
    if intent == "help":
        return help_message() + "\n\nLet's continue Q&A. Reply with your answer, or say skip/repeat."
    if intent == "status":
        return status_message(sess)
    if intent == "capabilities":
        return capabilities_message() + "\n\nWe can continue Q&A when you're ready."
    qna_questions = json.loads(sess["qna_questions"]) if sess["qna_questions"] else []
    idx = int(sess["qna_index"] or 0)

    # Clarification/deeper explanations
    if re.search(r"\bmore\s+details\b", text, re.I):
        title = sess["selected_paper_title"] or ""
        abstract = sess["selected_paper_abstract"] or ""
        if not title:
            return "No paper selected yet. Please search and select a paper first."
        # Section-specific details if requested
        sec_m = re.search(r"more\s+details\s+(intro|introduction|method|methodology|results|conclusions?)", text, re.I)
        if sec_m:
            section = sec_m.group(1).lower()
            sec_map = {"intro": "Introduction", "introduction": "Introduction", "method": "Methodology", "methodology": "Methodology", "results": "Results", "conclusion": "Conclusions", "conclusions": "Conclusions"}
            target = sec_map.get(section, "Introduction")
            
            # Check section cache
            section_cache_key = _cache_key(title, abstract, target)
            if section_cache_key in _section_cache:
                logger.debug(f"Cache hit for section: {section_cache_key}")
                return f"## {target}\n{_section_cache[section_cache_key]}"
            
            # Llama-optimized prompt for section details
            prompt = f"""You are explaining a research paper section in detail.

SECTION TO EXPLAIN: {target}

CONSTRAINTS:
- Write exactly 6-8 sentences
- Focus ONLY on {target.lower()} aspects
- Be specific and technical but accessible
- Base explanation only on the abstract provided

PAPER:
Title: {title}
Abstract: {abstract}

{target.upper()} EXPLANATION:"""
            
            detail = groq_generate_text(prompt)
            if detail:
                _section_cache[section_cache_key] = detail
            else:
                detail = "Detailed section unavailable right now."
            return f"## {target}\n{detail}"
        else:
            detail = summarize_paper(title, abstract, url="", detailed=True)
            return detail

    # If questions list is empty, initialize from selected paper
    if not qna_questions:
        title = sess["selected_paper_title"] or ""
        abstract = sess["selected_paper_abstract"] or ""
        if not title:
            update_session(user_id, mode="browsing", qna_active=0)
            return "No paper selected yet. Search and select a paper first."
        # Prefer generating from a brief summary for better Q&A
        summary = summarize_paper(title, abstract, url="")
        qna_questions = generate_qna_items(title, summary)
        update_session(user_id, qna_questions=json.dumps(qna_questions), qna_index=0, qna_active=1)
        return f"Q&A started!\nQ1: {qna_questions[0]['q']}"

    # Evaluate current question
    if idx < 0 or idx >= len(qna_questions):
        update_session(user_id, mode="browsing", qna_active=0)
        return "Q&A finished! Great job. You're back to browsing."

    # Q&A controls
    if intent == "qna_repeat":
        return f"Repeating:\nQ{idx+1}: {qna_questions[idx]['q']}"
    if intent == "qna_skip":
        idx += 1
        if idx < len(qna_questions):
            update_session(user_id, qna_index=idx)
            return f"Skipped.\n\nQ{idx+1}: {qna_questions[idx]['q']}"
        else:
            update_session(
                user_id,
                mode="browsing",
                qna_active=0,
                qna_index=0,
                qna_questions=json.dumps([]),
            )
            return "Q&A finished! You're back to browsing."

    current = qna_questions[idx]
    score, feedback = evaluate_answer(text, current.get("a_keywords", []))
    total = int(sess["score"] or 0) + score
    idx += 1
    if idx < len(qna_questions):
        update_session(user_id, qna_index=idx, score=total)
        return f"{feedback}\n\nQ{idx+1}: {qna_questions[idx]['q']}"
    else:
        # Cleanup selected paper to avoid persisting beyond session
        update_session(
            user_id,
            mode="browsing",
            qna_active=0,
            qna_index=0,
            qna_questions=json.dumps([]),
            score=total,
            selected_paper_id=None,
            selected_paper_title=None,
            selected_paper_abstract=None,
        )
        return f"{feedback}\n\nQ&A complete! Your score: {total}. You're back to browsing."


# ---------------------------
# Flask routes
# ---------------------------

@app.route("/")
def root() -> str:
    return "Research Paper WhatsApp Bot is running."


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {}
    }
    
    # Check 1: Database connectivity
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["checks"]["database"] = f"error: {str(e)}"
        health["status"] = "degraded"
    
    # Check 2: Groq API key configured
    health["checks"]["groq_configured"] = "ok" if GROQ_API_KEY else "missing"
    if not GROQ_API_KEY:
        health["status"] = "degraded"
    
    # Check 3: Twilio configured
    health["checks"]["twilio_configured"] = "ok" if (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN) else "missing"
    
    status_code = 200 if health["status"] == "healthy" else 503
    return json.dumps(health), status_code, {"Content-Type": "application/json"}


@app.route("/whatsapp", methods=["POST"])
@validate_twilio_request
def whatsapp_webhook():
    init_db()  # ensure tables exist
    from_number = request.form.get("From", "")
    body = request.form.get("Body", "").strip()
    user_id = from_number or "unknown"

    log_message(user_id, "user", body)
    sess = get_session(user_id)

    # Mode transitions
    if is_qna_start(body):
        update_session(user_id, mode="qna", qna_active=1, qna_index=0, qna_questions=json.dumps([]))
        reply_text = handle_qna(user_id, body)
    else:
        mode = sess["mode"] or "browsing"
        if mode == "qna":
            reply_text = handle_qna(user_id, body)
        else:
            reply_text = handle_browsing(user_id, body)

    log_message(user_id, "bot", reply_text)

    # Split into chunks with continuation tags
    def split_chunks(text: str, limit: int = 1500) -> List[str]:
        lines = []
        while text:
            chunk = text[:limit]
            text = text[limit:]
            lines.append(chunk)
        # add (1/2) tags if multiple chunks
        if len(lines) > 1:
            tagged = []
            total = len(lines)
            for i, c in enumerate(lines, 1):
                tagged.append(f"({i}/{total})\n{c}")
            return tagged
        return lines

    resp = MessagingResponse()
    for part in split_chunks(reply_text):
        resp.message(part)
    return str(resp)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
