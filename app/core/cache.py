"""
Redis cache management
"""

import json
from typing import Optional, Any
import redis.asyncio as redis

from .config import settings, logger


class CacheManager:
    """Redis cache manager with fallback to in-memory cache"""

    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self.local_cache: dict = {}  # Fallback in-memory cache

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.warning(f"Redis unavailable, using local cache: {e}")
            self.redis = None

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        if self.redis:
            try:
                return await self.redis.get(key)
            except Exception as e:
                logger.error(f"Redis get error: {e}")

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
            except Exception as e:
                logger.error(f"Redis set error: {e}")

        self.local_cache[key] = value

    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
        return None

    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set JSON value"""
        try:
            json_str = json.dumps(value)
            await self.set(key, json_str, ttl)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON encode error: {e}")

    async def delete(self, key: str):
        """Delete cached value"""
        if self.redis:
            try:
                await self.redis.delete(key)
                return
            except Exception as e:
                logger.error(f"Redis delete error: {e}")

        self.local_cache.pop(key, None)

    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        if self.redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor,
                        match=pattern,
                        count=100
                    )
                    if keys:
                        await self.redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.error(f"Redis clear pattern error: {e}")


# Global cache instance
cache = CacheManager()


__all__ = ["cache", "CacheManager"]
