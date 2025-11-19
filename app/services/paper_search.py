"""
Paper search service - Multi-source paper retrieval
Supports Semantic Scholar and arXiv APIs
"""

import re
import httpx
from typing import List, Dict, Optional
from app.core.config import logger

# Shared HTTP client
http_client = httpx.AsyncClient(timeout=30.0)


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


async def search_papers_combined(query: str, limit: int = 5) -> List[Dict]:
    """Search both Semantic Scholar and arXiv, return combined results"""
    # Search both in parallel
    ss_results, arxiv_results = await asyncio.gather(
        search_papers_semantic_scholar(query, limit=limit//2 + 1),
        search_papers_arxiv(query, limit=limit//2 + 1)
    )

    # Combine and deduplicate by title
    seen_titles = set()
    combined = []

    for paper in ss_results + arxiv_results:
        title_normalized = paper.get("title", "").lower().strip()
        if title_normalized and title_normalized not in seen_titles:
            seen_titles.add(title_normalized)
            combined.append(paper)

            if len(combined) >= limit:
                break

    return combined


async def get_paper_by_doi(doi: str) -> Optional[Dict]:
    """Get paper metadata by DOI"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,authors,year,url,abstract,citationCount,venue"}

    try:
        response = await http_client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "paperId": data.get("paperId"),
            "title": data.get("title"),
            "year": data.get("year"),
            "url": data.get("url"),
            "authors": ", ".join([a.get("name", "") for a in data.get("authors", []) if a]),
            "abstract": data.get("abstract") or "",
            "citationCount": data.get("citationCount", 0),
            "venue": data.get("venue", ""),
        }
    except Exception as e:
        logger.error(f"DOI lookup failed: {e}")
        return None


async def get_paper_by_arxiv_id(arxiv_id: str) -> Optional[Dict]:
    """Get paper metadata by arXiv ID"""
    base = "http://export.arxiv.org/api/query"
    params = {"id_list": arxiv_id}

    try:
        response = await http_client.get(base, params=params)
        response.raise_for_status()
        text = response.text

        if "<entry>" not in text:
            return None

        e = text.split("<entry>")[1].split("</entry>")[0]

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

        return {
            "paperId": arxiv_id,
            "title": title,
            "year": year,
            "url": link,
            "authors": authors,
            "abstract": summary,
            "citationCount": 0,
            "venue": "arXiv",
        }
    except Exception as e:
        logger.error(f"arXiv lookup failed: {e}")
        return None


# Import asyncio at the top
import asyncio


__all__ = [
    "search_papers_semantic_scholar",
    "search_papers_arxiv",
    "search_papers_combined",
    "get_paper_by_doi",
    "get_paper_by_arxiv_id",
    "http_client"
]
