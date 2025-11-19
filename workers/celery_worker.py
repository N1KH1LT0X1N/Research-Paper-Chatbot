"""
Celery worker for background tasks
- PDF download and processing
- Embedding generation
- Daily paper notifications
- Spaced repetition reminders
"""

import os
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv()

# Initialize Celery
celery_app = Celery(
    'research_bot_worker',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

# Configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
)

# Periodic tasks schedule
celery_app.conf.beat_schedule = {
    'send-daily-papers': {
        'task': 'celery_worker.send_daily_papers',
        'schedule': crontab(hour=9, minute=0),  # 9 AM daily
    },
    'send-review-reminders': {
        'task': 'celery_worker.send_review_reminders',
        'schedule': crontab(hour=10, minute=0),  # 10 AM daily
    },
    'cleanup-old-cache': {
        'task': 'celery_worker.cleanup_cache',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    },
}


@celery_app.task(name='celery_worker.download_and_process_pdf')
def download_and_process_pdf(paper_id: str, arxiv_id: str):
    """
    Download PDF from arXiv and process it
    - Extract full text
    - Extract figures and tables
    - Generate embeddings
    - Store in database
    """
    import asyncio
    from async_research_bot import (
        download_pdf_from_arxiv,
        extract_pdf_text,
        async_session_maker,
        Paper,
        generate_embedding
    )
    from sqlalchemy import update as sql_update

    async def process():
        try:
            # Download PDF
            pdf_bytes = await download_pdf_from_arxiv(arxiv_id)
            if not pdf_bytes:
                return {"status": "failed", "reason": "download_failed"}

            # Extract text
            extracted = await extract_pdf_text(pdf_bytes)

            # Generate embedding from full text
            full_text = extracted.get("full_text", "")
            embedding = None
            if full_text:
                embedding = await generate_embedding(full_text[:5000])  # Use first 5k chars

            # Update database
            async with async_session_maker() as db:
                await db.execute(
                    sql_update(Paper).where(Paper.paper_id == paper_id).values(
                        full_text=full_text,
                        sections=extracted.get("sections", {}),
                        embeddings=embedding.tolist() if embedding is not None else None
                    )
                )
                await db.commit()

            return {
                "status": "success",
                "page_count": extracted.get("page_count", 0),
                "sections": len(extracted.get("sections", {}))
            }

        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    # Run async function
    return asyncio.run(process())


@celery_app.task(name='celery_worker.generate_paper_embedding')
def generate_paper_embedding(paper_id: str, title: str, abstract: str):
    """Generate and store embedding for a paper"""
    import asyncio
    from async_research_bot import store_paper_embedding

    async def process():
        await store_paper_embedding(
            paper_id,
            title,
            abstract,
            {"title": title}
        )
        return {"status": "success"}

    return asyncio.run(process())


@celery_app.task(name='celery_worker.send_daily_papers')
def send_daily_papers():
    """Send daily paper recommendations to active users"""
    import asyncio
    from async_research_bot import (
        async_session_maker,
        Session,
        recommend_papers,
        twilio_client,
        TWILIO_WHATSAPP_FROM
    )
    from sqlalchemy import select
    from datetime import datetime, timedelta

    async def process():
        async with async_session_maker() as db:
            # Get users active in last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)

            result = await db.execute(
                select(Session.user_id).where(
                    Session.updated_at >= week_ago
                ).distinct()
            )

            active_users = [row[0] for row in result]

            sent_count = 0
            for user_id in active_users:
                try:
                    # Get recommendations
                    papers = await recommend_papers(user_id, db, n=1)
                    if papers:
                        paper = papers[0]

                        message = f"""☀️ *Good Morning!*

Today's featured paper:

📄 *{paper['title']}*
👤 {paper.get('authors', '')[:50]}...
📅 {paper.get('year', '?')}

Would you like to read it? Reply 'select 1' to start!"""

                        # Send via Twilio
                        twilio_client.messages.create(
                            from_=TWILIO_WHATSAPP_FROM,
                            to=user_id,
                            body=message
                        )

                        sent_count += 1

                except Exception as e:
                    print(f"Failed to send to {user_id}: {e}")

            return {"status": "success", "sent": sent_count}

    return asyncio.run(process())


@celery_app.task(name='celery_worker.send_review_reminders')
def send_review_reminders():
    """Send spaced repetition review reminders"""
    import asyncio
    from async_research_bot import (
        async_session_maker,
        ReviewSchedule,
        Paper,
        twilio_client,
        TWILIO_WHATSAPP_FROM
    )
    from sqlalchemy import select
    from datetime import datetime

    async def process():
        async with async_session_maker() as db:
            # Get due reviews
            result = await db.execute(
                select(ReviewSchedule).where(
                    ReviewSchedule.next_review <= datetime.utcnow()
                )
            )

            due_reviews = result.scalars().all()

            # Group by user
            user_reviews = {}
            for review in due_reviews:
                if review.user_id not in user_reviews:
                    user_reviews[review.user_id] = []
                user_reviews[review.user_id].append(review)

            sent_count = 0
            for user_id, reviews in user_reviews.items():
                try:
                    count = len(reviews)

                    message = f"""📚 *Review Time!*

You have {count} question(s) due for review today.

Reviewing helps you remember what you've learned! 🧠

Type 'start review' to begin."""

                    twilio_client.messages.create(
                        from_=TWILIO_WHATSAPP_FROM,
                        to=user_id,
                        body=message
                    )

                    sent_count += 1

                except Exception as e:
                    print(f"Failed to send reminder to {user_id}: {e}")

            return {"status": "success", "users_notified": sent_count}

    return asyncio.run(process())


@celery_app.task(name='celery_worker.cleanup_cache')
def cleanup_cache():
    """Clean up old cache entries"""
    import redis
    import os

    try:
        r = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

        # Delete search results older than 24 hours
        # (Already handled by TTL in Redis, this is just for manual cleanup)

        # Count keys
        key_count = r.dbsize()

        return {"status": "success", "total_keys": key_count}

    except Exception as e:
        return {"status": "failed", "reason": str(e)}


@celery_app.task(name='celery_worker.build_citation_graph')
def build_citation_graph(paper_id: str, depth: int = 2):
    """Build citation graph for a paper (expensive operation)"""
    import asyncio
    import httpx
    from async_research_bot import async_session_maker, Paper
    from sqlalchemy import update as sql_update

    async def fetch_citations(pid: str):
        """Fetch citations from Semantic Scholar"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{pid}"
        params = {"fields": "citations.paperId,citations.title,references.paperId,references.title"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except:
                return {}

    async def process():
        try:
            # Fetch citation data
            data = await fetch_citations(paper_id)

            citations = data.get('citations', [])
            references = data.get('references', [])

            # Build graph structure
            graph = {
                'nodes': {},
                'edges': [],
                'stats': {
                    'citation_count': len(citations),
                    'reference_count': len(references)
                }
            }

            # Store in database metadata
            async with async_session_maker() as db:
                await db.execute(
                    sql_update(Paper).where(Paper.paper_id == paper_id).values(
                        metadata={'citation_graph': graph}
                    )
                )
                await db.commit()

            return {"status": "success", "citations": len(citations), "references": len(references)}

        except Exception as e:
            return {"status": "failed", "reason": str(e)}

    return asyncio.run(process())


if __name__ == '__main__':
    celery_app.start()
