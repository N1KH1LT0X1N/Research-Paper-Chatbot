"""Quick test script to verify database tables"""
import asyncio
from sqlalchemy import inspect
from app.core.database import engine


async def check_tables():
    async with engine.begin() as conn:
        tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        print(f'✅ Tables created ({len(tables)}):')
        for table in sorted(tables):
            print(f'   - {table}')


if __name__ == "__main__":
    asyncio.run(check_tables())
