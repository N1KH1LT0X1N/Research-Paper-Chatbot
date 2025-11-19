"""
Analytics service - User statistics, streaks, and recommendations
"""

from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.models import UserHistory, Achievement, Paper
from app.core.config import logger


async def generate_user_stats(user_id: str, db: AsyncSession) -> str:
    """Generate comprehensive user statistics"""

    # Count papers read
    result = await db.execute(
        select(func.count(UserHistory.id)).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "read"
        )
    )
    papers_read = result.scalar() or 0

    # Count Q&As completed and average score
    result = await db.execute(
        select(
            func.count(UserHistory.id),
            func.avg(UserHistory.score)
        ).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "qna_completed"
        )
    )
    row = result.first()
    qnas_completed = row[0] or 0
    avg_score = row[1] or 0

    # Count achievements
    result = await db.execute(
        select(func.count(Achievement.id)).where(
            Achievement.user_id == user_id
        )
    )
    achievement_count = result.scalar() or 0

    # Calculate streak
    streak = await calculate_streak(user_id, db)

    # Format message
    msg = f"""📊 *Your Statistics*

📖 Papers read: {papers_read}
✅ Q&As completed: {qnas_completed}
🎯 Average score: {avg_score:.1f}/10
🔥 Current streak: {streak} days
🏆 Achievements: {achievement_count}

Keep learning! 🚀"""

    return msg


async def calculate_streak(user_id: str, db: AsyncSession) -> int:
    """Calculate current study streak in days"""

    # Get distinct days with activity
    result = await db.execute(
        select(func.date(UserHistory.timestamp).label('day')).where(
            UserHistory.user_id == user_id
        ).group_by('day')
        .order_by(func.date(UserHistory.timestamp).desc())
        .limit(30)
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
    """
    Recommend papers based on user history

    Args:
        user_id: User identifier
        db: Database session
        n: Number of recommendations to return

    Returns:
        List of recommended papers
    """
    from app.services.paper_search import search_papers_semantic_scholar

    # Get user's reading history
    result = await db.execute(
        select(UserHistory.paper_id).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "read"
        ).limit(10)
    )
    read_paper_ids = [row[0] for row in result if row[0]]

    if not read_paper_ids:
        # Return trending papers for new users
        logger.info(f"No history for user {user_id}, returning trending papers")
        return await search_papers_semantic_scholar("machine learning", limit=n)

    # Get papers from history
    result = await db.execute(
        select(Paper).where(Paper.paper_id.in_(read_paper_ids))
    )
    read_papers = list(result.scalars().all())

    if not read_papers:
        return await search_papers_semantic_scholar("artificial intelligence", limit=n)

    # Extract keywords from read papers
    keywords = []
    for paper in read_papers:
        if paper.title:
            keywords.extend(paper.title.split()[:3])

    # Search for similar papers
    query = " ".join(keywords[:10])
    recommendations = await search_papers_semantic_scholar(query, limit=n + len(read_paper_ids))

    # Filter out already read papers
    filtered = [
        p for p in recommendations
        if p.get("paperId") not in read_paper_ids
    ]

    return filtered[:n]


async def get_activity_summary(user_id: str, db: AsyncSession, days: int = 7) -> Dict:
    """
    Get user activity summary for the last N days

    Args:
        user_id: User identifier
        db: Database session
        days: Number of days to look back

    Returns:
        Dictionary with activity metrics
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    # Get activity counts by type
    result = await db.execute(
        select(
            UserHistory.action,
            func.count(UserHistory.id)
        ).where(
            UserHistory.user_id == user_id,
            UserHistory.timestamp >= cutoff
        ).group_by(UserHistory.action)
    )

    activity = {row[0]: row[1] for row in result}

    return {
        "period_days": days,
        "papers_searched": activity.get("search", 0),
        "papers_read": activity.get("read", 0),
        "qnas_completed": activity.get("qna_completed", 0),
        "total_actions": sum(activity.values())
    }


async def award_achievement(
    user_id: str,
    achievement_type: str,
    description: str,
    db: AsyncSession
) -> bool:
    """
    Award an achievement to a user

    Args:
        user_id: User identifier
        achievement_type: Type of achievement
        description: Achievement description
        db: Database session

    Returns:
        True if newly awarded, False if already had it
    """
    # Check if already has this achievement
    result = await db.execute(
        select(Achievement).where(
            Achievement.user_id == user_id,
            Achievement.achievement_type == achievement_type
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return False

    # Award new achievement
    achievement = Achievement(
        user_id=user_id,
        achievement_type=achievement_type,
        description=description,
        earned_at=datetime.utcnow()
    )
    db.add(achievement)
    await db.commit()

    logger.info(f"Awarded achievement to {user_id}: {achievement_type}")
    return True


async def check_and_award_achievements(user_id: str, db: AsyncSession) -> List[str]:
    """
    Check and award applicable achievements

    Returns:
        List of newly awarded achievement descriptions
    """
    new_achievements = []

    # Get stats
    result = await db.execute(
        select(func.count(UserHistory.id)).where(
            UserHistory.user_id == user_id,
            UserHistory.action == "read"
        )
    )
    papers_read = result.scalar() or 0

    # First paper
    if papers_read == 1:
        if await award_achievement(user_id, "first_paper", "Read your first paper", db):
            new_achievements.append("🎉 First Paper!")

    # 10 papers
    if papers_read >= 10:
        if await award_achievement(user_id, "bibliophile", "Read 10 papers", db):
            new_achievements.append("📚 Bibliophile - 10 papers read!")

    # 50 papers
    if papers_read >= 50:
        if await award_achievement(user_id, "scholar", "Read 50 papers", db):
            new_achievements.append("🎓 Scholar - 50 papers read!")

    # Streak achievements
    streak = await calculate_streak(user_id, db)
    if streak >= 7:
        if await award_achievement(user_id, "week_streak", "7-day streak", db):
            new_achievements.append("🔥 Week Warrior - 7 day streak!")

    if streak >= 30:
        if await award_achievement(user_id, "month_streak", "30-day streak", db):
            new_achievements.append("💪 Monthly Master - 30 day streak!")

    return new_achievements


__all__ = [
    "generate_user_stats",
    "calculate_streak",
    "recommend_papers",
    "get_activity_summary",
    "award_achievement",
    "check_and_award_achievements"
]
