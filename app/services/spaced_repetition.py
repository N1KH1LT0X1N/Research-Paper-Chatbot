"""
Spaced repetition service - SM-2 algorithm for review scheduling
"""

from datetime import datetime, timedelta
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import ReviewSchedule
from app.core.config import logger


async def schedule_review(
    user_id: str,
    paper_id: str,
    question_id: str,
    performance: float,
    db: AsyncSession
):
    """
    Schedule next review using SM-2 algorithm

    Args:
        user_id: User identifier
        paper_id: Paper identifier
        question_id: Question identifier
        performance: Score from 0-10
        db: Database session
    """

    # Get existing review record
    result = await db.execute(
        select(ReviewSchedule).where(
            ReviewSchedule.user_id == user_id,
            ReviewSchedule.question_id == question_id
        )
    )
    review = result.scalar_one_or_none()

    # Create new review record if doesn't exist
    if not review:
        review = ReviewSchedule(
            user_id=user_id,
            paper_id=paper_id,
            question_id=question_id,
            easiness=2.5,  # Default easiness factor
            interval=1,    # Start with 1 day
            repetitions=0
        )
        db.add(review)

    # SM-2 algorithm
    if performance >= 6:  # Correct answer threshold
        if review.repetitions == 0:
            interval = 1  # First repetition: 1 day
        elif review.repetitions == 1:
            interval = 6  # Second repetition: 6 days
        else:
            # Subsequent repetitions: multiply by easiness
            interval = int(review.interval * review.easiness)
        repetitions = review.repetitions + 1
    else:  # Incorrect answer
        interval = 1  # Reset to 1 day
        repetitions = 0  # Reset repetition count

    # Update easiness factor
    # Formula: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    # where q = performance/2 (scale from 0-10 to 0-5)
    q = performance / 2
    easiness = review.easiness + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    easiness = max(1.3, easiness)  # Minimum easiness factor

    # Update review record
    review.easiness = easiness
    review.interval = interval
    review.repetitions = repetitions
    review.next_review = datetime.utcnow() + timedelta(days=interval)
    review.last_score = performance

    await db.commit()
    logger.info(f"Scheduled review for user {user_id}, question {question_id}: "
                f"interval={interval} days, easiness={easiness:.2f}")


async def get_due_reviews(user_id: str, db: AsyncSession) -> List[ReviewSchedule]:
    """
    Get questions due for review

    Args:
        user_id: User identifier
        db: Database session

    Returns:
        List of ReviewSchedule objects due for review
    """
    result = await db.execute(
        select(ReviewSchedule).where(
            ReviewSchedule.user_id == user_id,
            ReviewSchedule.next_review <= datetime.utcnow()
        )
    )
    return list(result.scalars().all())


async def get_review_stats(user_id: str, db: AsyncSession) -> dict:
    """
    Get user's review statistics

    Args:
        user_id: User identifier
        db: Database session

    Returns:
        Dictionary with review statistics
    """
    result = await db.execute(
        select(ReviewSchedule).where(ReviewSchedule.user_id == user_id)
    )
    reviews = list(result.scalars().all())

    if not reviews:
        return {
            "total_reviews": 0,
            "due_now": 0,
            "average_easiness": 0,
            "mastered": 0
        }

    due_now = sum(1 for r in reviews if r.next_review <= datetime.utcnow())
    average_easiness = sum(r.easiness for r in reviews) / len(reviews)
    mastered = sum(1 for r in reviews if r.repetitions >= 3 and r.easiness >= 2.5)

    return {
        "total_reviews": len(reviews),
        "due_now": due_now,
        "average_easiness": round(average_easiness, 2),
        "mastered": mastered
    }


async def get_next_review_time(user_id: str, db: AsyncSession) -> datetime:
    """
    Get the next review time for a user

    Args:
        user_id: User identifier
        db: Database session

    Returns:
        Datetime of next review, or None if no reviews scheduled
    """
    result = await db.execute(
        select(ReviewSchedule)
        .where(ReviewSchedule.user_id == user_id)
        .order_by(ReviewSchedule.next_review)
        .limit(1)
    )
    review = result.scalar_one_or_none()

    return review.next_review if review else None


def calculate_retention_score(review: ReviewSchedule) -> float:
    """
    Calculate a retention score based on review performance

    Args:
        review: ReviewSchedule object

    Returns:
        Retention score from 0-100
    """
    # Base score on repetitions and easiness
    repetition_score = min(review.repetitions * 20, 60)  # Max 60 from repetitions
    easiness_score = min((review.easiness - 1.3) / 1.7 * 40, 40)  # Max 40 from easiness

    return repetition_score + easiness_score


__all__ = [
    "schedule_review",
    "get_due_reviews",
    "get_review_stats",
    "get_next_review_time",
    "calculate_retention_score"
]
