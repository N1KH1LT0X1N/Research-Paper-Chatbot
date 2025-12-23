"""
Q&A service - Question generation, answer grading, and semantic evaluation
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import logger
from app.services.ai_service import gemini_generate_text
from app.services.vector_search import generate_embedding, EMBEDDINGS_AVAILABLE

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False


async def generate_qna_items(
    title: str,
    source_text: str,
    difficulty: str = "medium",
    n: int = 3
) -> List[Dict[str, Any]]:
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

    # Fallback if AI generation fails
    if not items:
        base_kws = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", source_text)][:9]
        items = [
            {
                "q": f"What is the main contribution of '{title}'?",
                "a": "The main contribution is...",
                "a_keywords": base_kws[:3]
            },
            {
                "q": "Describe the methodology used.",
                "a": "The methodology involves...",
                "a_keywords": base_kws[3:6]
            },
            {
                "q": "What are the key results?",
                "a": "The key results show...",
                "a_keywords": base_kws[6:9]
            },
        ]

    return items[:n]


async def evaluate_answer_semantic(
    question: str,
    user_answer: str,
    reference_answer: str,
    keywords: List[str]
) -> Tuple[float, str]:
    """Semantic answer evaluation with detailed feedback"""

    scores = []

    # Method 1: Keyword matching (fast)
    keyword_score = sum(1 for k in keywords if k.lower() in user_answer.lower())
    keyword_ratio = keyword_score / max(len(keywords), 1)
    scores.append(keyword_ratio * 10)

    # Method 2: Embedding similarity (if available)
    if EMBEDDINGS_AVAILABLE and SIMILARITY_AVAILABLE:
        try:
            user_emb = await generate_embedding(user_answer)
            ref_emb = await generate_embedding(reference_answer)

            if user_emb is not None and ref_emb is not None:
                sim = cosine_similarity(
                    [user_emb],
                    [ref_emb]
                )[0][0]
                scores.append(sim * 10)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}")

    # Method 3: LLM grading (most accurate but slower)
    llm_score = await grade_with_llm(question, user_answer, reference_answer)
    if llm_score is not None:
        scores.append(llm_score)

    # Weighted average based on available methods
    if len(scores) >= 3:
        final_score = 0.2 * scores[0] + 0.3 * scores[1] + 0.5 * scores[2]
    elif len(scores) == 2:
        final_score = 0.4 * scores[0] + 0.6 * scores[1]
    else:
        final_score = scores[0] if scores else 5.0

    # Generate feedback
    feedback = await generate_feedback(final_score, keywords, keyword_score, user_answer)

    return final_score, feedback


async def grade_with_llm(
    question: str,
    user_answer: str,
    reference: str
) -> Optional[float]:
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
            score_match = re.search(r'\d+\.?\d*', response)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0), 10)
        except Exception as e:
            logger.warning(f"LLM grading parsing failed: {e}")
    return None


async def generate_feedback(
    score: float,
    keywords: List[str],
    keyword_count: int,
    user_answer: str
) -> str:
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


def calculate_answer_score_simple(
    user_answer: str,
    reference_answer: str,
    keywords: List[str]
) -> float:
    """Simple keyword-based scoring (synchronous fallback)"""
    user_lower = user_answer.lower()

    # Count keyword matches
    keyword_matches = sum(1 for k in keywords if k.lower() in user_lower)
    keyword_score = keyword_matches / max(len(keywords), 1)

    # Check if reference terms are present
    ref_words = set(reference_answer.lower().split())
    user_words = set(user_lower.split())
    overlap = len(ref_words & user_words) / max(len(ref_words), 1)

    # Combined score
    score = (keyword_score * 0.6 + overlap * 0.4) * 10

    return min(score, 10.0)


__all__ = [
    "generate_qna_items",
    "evaluate_answer_semantic",
    "grade_with_llm",
    "generate_feedback",
    "calculate_answer_score_simple"
]
