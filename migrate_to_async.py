"""
Database Migration Script
Migrate data from old SQLite database to new async database
"""

import asyncio
import sqlite3
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from dotenv import load_dotenv
import os

# Import models from new async app
from async_research_bot import (
    Base, Session as AsyncSession_Model, Paper, UserHistory,
    ChatLog, ReadingList, StudyGroup, Achievement, ReviewSchedule
)

load_dotenv()

# Old database
OLD_DB_PATH = os.getenv("OLD_DB_PATH", "whatsapp_bot.db")

# New database
NEW_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./whatsapp_bot_async.db")


async def migrate_data():
    """Main migration function"""

    print("=" * 60)
    print("DATABASE MIGRATION: SQLite (sync) → Async Database")
    print("=" * 60)

    # Check if old database exists
    if not os.path.exists(OLD_DB_PATH):
        print(f"❌ Old database not found at: {OLD_DB_PATH}")
        print("   Nothing to migrate. Starting fresh!")
        return

    # Connect to new database
    engine = create_async_engine(NEW_DATABASE_URL, echo=True)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create tables in new database
    print("\n1️⃣  Creating new database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Schema created")

    # Connect to old database
    print("\n2️⃣  Reading from old database...")
    old_conn = sqlite3.connect(OLD_DB_PATH)
    old_conn.row_factory = sqlite3.Row

    # Migrate sessions
    print("\n3️⃣  Migrating sessions...")
    await migrate_sessions(old_conn, async_session_maker)

    # Migrate logs
    print("\n4️⃣  Migrating chat logs...")
    await migrate_logs(old_conn, async_session_maker)

    # Close old connection
    old_conn.close()

    print("\n" + "=" * 60)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the new application: python async_research_bot.py")
    print("2. Verify data was migrated correctly")
    print(f"3. Backup old database: cp {OLD_DB_PATH} {OLD_DB_PATH}.backup")
    print("4. Update .env to point to new DATABASE_URL")


async def migrate_sessions(old_conn: sqlite3.Connection, async_session_maker):
    """Migrate user sessions"""

    try:
        old_sessions = old_conn.execute("SELECT * FROM sessions").fetchall()
    except sqlite3.OperationalError:
        print("   ⚠️  No sessions table in old database")
        return

    count = 0
    async with async_session_maker() as db:
        for row in old_sessions:
            # Map old columns to new schema
            new_session = AsyncSession_Model(
                user_id=row['user_id'],
                mode=row['mode'] or 'browsing',
                selected_paper_id=row['selected_paper_id'],
                selected_paper_title=row['selected_paper_title'],
                selected_paper_abstract=row['selected_paper_abstract'],
                qna_active=bool(row['qna_active']),
                qna_index=row['qna_index'] or 0,
                qna_questions=json.loads(row['qna_questions']) if row['qna_questions'] else None,
                score=row['score'] or 0,
                last_results=json.loads(row['last_results']) if row.get('last_results') else None,
                difficulty_preference='medium',
                updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else datetime.utcnow()
            )

            db.add(new_session)
            count += 1

        await db.commit()

    print(f"   ✅ Migrated {count} sessions")


async def migrate_logs(old_conn: sqlite3.Connection, async_session_maker):
    """Migrate chat logs"""

    try:
        old_logs = old_conn.execute("SELECT * FROM logs ORDER BY id LIMIT 10000").fetchall()
    except sqlite3.OperationalError:
        print("   ⚠️  No logs table in old database")
        return

    count = 0
    async with async_session_maker() as db:
        for row in old_logs:
            new_log = ChatLog(
                user_id=row['user_id'],
                role=row['role'],
                message=row['message'],
                created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else datetime.utcnow()
            )

            db.add(new_log)
            count += 1

            # Commit in batches
            if count % 1000 == 0:
                await db.commit()
                print(f"   ... {count} logs migrated")

        await db.commit()

    print(f"   ✅ Migrated {count} chat logs")


async def create_fresh_database():
    """Create fresh database without migration"""

    print("=" * 60)
    print("CREATING FRESH DATABASE")
    print("=" * 60)

    engine = create_async_engine(NEW_DATABASE_URL, echo=True)

    print("\nCreating database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("\n✅ Fresh database created!")
    print(f"\nDatabase: {NEW_DATABASE_URL}")


async def verify_migration():
    """Verify migration was successful"""

    print("\n" + "=" * 60)
    print("VERIFYING MIGRATION")
    print("=" * 60)

    engine = create_async_engine(NEW_DATABASE_URL, echo=False)
    async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_maker() as db:
        # Count sessions
        result = await db.execute(select(AsyncSession_Model))
        sessions = result.scalars().all()
        print(f"\n✅ Sessions: {len(sessions)}")

        # Count logs
        result = await db.execute(select(ChatLog))
        logs = result.scalars().all()
        print(f"✅ Chat logs: {len(logs)}")

        # Show sample session
        if sessions:
            sample = sessions[0]
            print(f"\nSample session:")
            print(f"  User: {sample.user_id}")
            print(f"  Mode: {sample.mode}")
            print(f"  Last paper: {sample.selected_paper_title}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--fresh":
        # Create fresh database
        asyncio.run(create_fresh_database())
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Verify migration
        asyncio.run(verify_migration())
    else:
        # Run migration
        asyncio.run(migrate_data())
