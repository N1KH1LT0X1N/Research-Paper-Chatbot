"""
Message handler service - Main business logic for processing user messages
"""

import re
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

from app.core.config import logger
from app.core.database import get_session, update_session, log_message
from app.models import UserHistory

from app.services import (
    search_papers_combined,
    summarize_paper,
    generate_qna_items,
    evaluate_answer_semantic,
    generate_user_stats,
    recommend_papers,
    check_and_award_achievements
)


def detect_intent(text: str) -> str:
    """Detect user intent from message"""
    text_lower = text.lower()

    # Help and commands
    if re.search(r'\b(help|commands?)\b', text_lower):
        return "help"
    if re.search(r'\b(status|where am i)\b', text_lower):
        return "status"
    if re.search(r'\b(reset|clear)\b', text_lower):
        return "reset"

    # Stats and analytics
    if re.search(r'\b(my stats|statistics|analytics)\b', text_lower):
        return "stats"
    if re.search(r'\brecommend\b', text_lower):
        return "recommend"

    # Q&A
    if re.search(r'\b(ready for q.?a|start qna|let.?s do q.?a)\b', text_lower):
        return "qna_start"
    if re.search(r'\bskip\b', text_lower):
        return "qna_skip"
    if re.search(r'\brepeat\b', text_lower):
        return "qna_repeat"

    # Paper selection
    if re.search(r'\b(select|choose|pick)\s+\d+\b', text_lower):
        return "selection"

    # Summary request
    if re.search(r'\b(summary|summarize)\b', text_lower):
        return "summary"

    # Paper search (URLs, DOIs, or keywords)
    if re.search(r'https?://|arxiv|\d{4}\.\d{4,5}|10\.\d{4,}/', text):
        return "paper"

    # Content-based search (3+ meaningful words)
    words = [
        w for w in re.findall(r'\b\w{3,}\b', text_lower)
        if w not in {'the', 'is', 'a', 'an', 'of', 'and', 'or', 'to', 'in'}
    ]
    if len(words) >= 3:
        return "paper"

    return "ambiguous"


async def handle_paper_search(
    user_id: str,
    query: str,
    db: AsyncSession
) -> str:
    """Handle paper search queries"""
    try:
        # Search for papers
        results = await search_papers_combined(query, limit=5)

        if not results:
            return "🔍 No papers found. Try different keywords or a broader query."

        # Store search in history
        history = UserHistory(
            user_id=user_id,
            action="search",
            query=query,
            timestamp=datetime.utcnow()
        )
        db.add(history)
        await db.commit()

        # Format results
        response = f"📚 *Found {len(results)} papers:*\n\n"

        for i, paper in enumerate(results, 1):
            title = paper.get("title", "Untitled")[:80]
            authors = paper.get("authors", "Unknown")[:50]
            year = paper.get("year", "")
            citations = paper.get("citationCount", 0)

            response += f"*{i}.* {title}\n"
            response += f"   👥 {authors}"
            if year:
                response += f" ({year})"
            if citations:
                response += f"\n   📊 {citations} citations"
            response += "\n\n"

        response += "Reply '*select <number>*' to view a paper summary."

        # Update session with search results
        await update_session(
            user_id, db,
            mode="browsing",
            last_search_results=results
        )

        return response

    except Exception as e:
        logger.error(f"Paper search failed: {e}")
        return "❌ Search failed. Please try again."


async def handle_paper_selection(
    user_id: str,
    selection_num: int,
    db: AsyncSession
) -> str:
    """Handle paper selection and generate summary"""
    try:
        # Get session
        session = await get_session(user_id, db)

        if not session.last_search_results:
            return "Please search for papers first using keywords or paper title."

        if selection_num < 1 or selection_num > len(session.last_search_results):
            return f"Invalid selection. Please choose 1-{len(session.last_search_results)}."

        paper = session.last_search_results[selection_num - 1]

        # Generate summary
        summary = await summarize_paper(
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            detailed=False
        )

        # Log to history
        history = UserHistory(
            user_id=user_id,
            action="read",
            paper_id=paper.get("paperId"),
            timestamp=datetime.utcnow()
        )
        db.add(history)

        # Update session
        await update_session(
            user_id, db,
            selected_paper_id=paper.get("paperId"),
            selected_paper_title=paper.get("title"),
            selected_paper_abstract=paper.get("abstract")
        )

        await db.commit()

        # Check for achievements
        achievements = await check_and_award_achievements(user_id, db)

        # Format response
        title = paper.get("title", "Untitled")
        url = paper.get("url", "")

        response = f"📄 *{title}*\n\n{summary}\n\n🔗 {url}"

        if achievements:
            response += "\n\n🎉 " + " ".join(achievements)

        response += "\n\nType '*start qna*' to test your understanding!"

        return response

    except Exception as e:
        logger.error(f"Paper selection failed: {e}")
        return "❌ Failed to load paper. Please try again."


async def handle_qna_start(user_id: str, db: AsyncSession) -> str:
    """Start Q&A session"""
    try:
        session = await get_session(user_id, db)

        if not session.selected_paper_title or not session.selected_paper_abstract:
            return "Please select a paper first before starting Q&A."

        # Generate questions
        questions = await generate_qna_items(
            title=session.selected_paper_title,
            source_text=session.selected_paper_abstract[:1500],
            difficulty="medium",
            n=3
        )

        if not questions:
            return "❌ Failed to generate questions. Please try again."

        # Store questions in session
        await update_session(
            user_id, db,
            mode="qna",
            qna_active=True,
            qna_questions=questions,
            qna_current_index=0,
            qna_score=0.0
        )

        # Return first question
        q = questions[0]
        return f"""🧠 *Q&A Mode Active*

*Question 1/3:*
{q['q']}

Reply with your answer!"""

    except Exception as e:
        logger.error(f"QNA start failed: {e}")
        return "❌ Failed to start Q&A. Please try again."


async def handle_qna_answer(user_id: str, answer: str, db: AsyncSession) -> str:
    """Handle Q&A answer"""
    try:
        session = await get_session(user_id, db)

        if not session.qna_active or not session.qna_questions:
            return "No active Q&A session. Type '*start qna*' to begin."

        idx = session.qna_current_index or 0
        questions = session.qna_questions

        if idx >= len(questions):
            return "Q&A session completed! Type '*start qna*' to try again."

        q_item = questions[idx]

        # Evaluate answer
        score, feedback = await evaluate_answer_semantic(
            question=q_item['q'],
            user_answer=answer,
            reference_answer=q_item['a'],
            keywords=q_item.get('a_keywords', [])
        )

        # Update total score
        total_score = (session.qna_score or 0) + score

        # Move to next question
        next_idx = idx + 1

        if next_idx < len(questions):
            # More questions remain
            await update_session(
                user_id, db,
                qna_current_index=next_idx,
                qna_score=total_score
            )

            next_q = questions[next_idx]
            response = f"""{feedback}

*Score:* {score:.1f}/10

---

*Question {next_idx + 1}/{len(questions)}:*
{next_q['q']}"""

        else:
            # Q&A completed
            avg_score = total_score / len(questions)

            # Log completion
            history = UserHistory(
                user_id=user_id,
                action="qna_completed",
                paper_id=session.selected_paper_id,
                score=avg_score,
                timestamp=datetime.utcnow()
            )
            db.add(history)

            await update_session(
                user_id, db,
                qna_active=False,
                mode="browsing"
            )

            await db.commit()

            response = f"""{feedback}

*Score:* {score:.1f}/10

---

✅ *Q&A Complete!*

*Final Average:* {avg_score:.1f}/10

{"🌟 Excellent work!" if avg_score >= 8 else "👍 Good job!" if avg_score >= 6 else "💪 Keep practicing!"}"""

        return response

    except Exception as e:
        logger.error(f"QNA answer handling failed: {e}")
        return "❌ Failed to process answer. Please try again."


async def handle_message(user_id: str, text: str, db: AsyncSession) -> str:
    """Main message handler - routes to appropriate handler"""

    session = await get_session(user_id, db)
    intent = detect_intent(text)

    # Special case: if in QNA mode and not a command, treat as answer
    if session.qna_active and intent not in ["help", "status", "reset", "qna_skip"]:
        return await handle_qna_answer(user_id, text, db)

    # Help
    if intent == "help":
        return """📚 *Research Paper Bot*

*Search:* Send paper title or keywords
*Select:* Reply 'select 1' after search
*Summary:* Get paper summaries
*Q&A:* Type 'start qna' to test understanding
*Stats:* Type 'my stats' for analytics
*Recommend:* Get personalized recommendations

Commands: help | status | reset"""

    # Status
    if intent == "status":
        mode = session.mode or "browsing"
        paper = session.selected_paper_title or "None"
        qna = "Active" if session.qna_active else "Inactive"
        return f"""📍 *Status*

Mode: {mode}
Paper: {paper[:50]}
Q&A: {qna}"""

    # Reset
    if intent == "reset":
        await update_session(
            user_id, db,
            mode="browsing",
            qna_active=False,
            selected_paper_id=None,
            selected_paper_title=None
        )
        return "✨ Session reset! Send a paper title to search."

    # Stats
    if intent == "stats":
        return await generate_user_stats(user_id, db)

    # Recommend
    if intent == "recommend":
        papers = await recommend_papers(user_id, db, n=3)
        if not papers:
            return "No recommendations yet. Read some papers first!"

        response = "🎯 *Recommended Papers:*\n\n"
        for i, p in enumerate(papers, 1):
            response += f"{i}. {p.get('title', 'Untitled')[:70]}\n"
        return response

    # Q&A start
    if intent == "qna_start":
        return await handle_qna_start(user_id, db)

    # Paper selection
    if intent == "selection":
        match = re.search(r'(\d+)', text)
        if match:
            num = int(match.group(1))
            return await handle_paper_selection(user_id, num, db)

    # Paper search
    if intent == "paper":
        return await handle_paper_search(user_id, text, db)

    # Ambiguous
    return """🤔 I didn't understand that. Try:
- Searching: 'machine learning transformers'
- Selecting: 'select 1'
- Q&A: 'start qna'
- Help: 'help'"""


__all__ = ["handle_message", "detect_intent"]
