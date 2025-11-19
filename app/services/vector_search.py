"""
Vector search service - Embeddings and RAG with SPECTER2 and ChromaDB
"""

import asyncio
from typing import Optional, Dict, List, Any
from app.core.config import logger

# Import embeddings with fallback
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
    # Initialize SPECTER2 model for scientific papers
    embedding_model = SentenceTransformer('allenai/specter2')
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    embedding_model = None
    np = None  # Define np as None for type hints
    logger.warning("Embeddings libraries not available")

# Import vector store with fallback
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    VECTOR_STORE_AVAILABLE = True

    # Initialize ChromaDB client
    chroma_client = chromadb.Client(ChromaSettings(
        anonymized_telemetry=False,
        allow_reset=True
    ))

    # Get or create collection
    vector_store = chroma_client.get_or_create_collection(
        name="research_papers",
        metadata={"description": "Research paper embeddings"}
    )
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    chroma_client = None
    vector_store = None
    logger.warning("ChromaDB not available")


async def generate_embedding(text: str) -> Optional[Any]:
    """Generate embedding for text using SPECTER2"""
    if not embedding_model or not EMBEDDINGS_AVAILABLE:
        return None

    try:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: embedding_model.encode(text, convert_to_numpy=True)
        )
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


async def store_paper_embedding(
    paper_id: str,
    title: str,
    abstract: str,
    metadata: Dict
) -> bool:
    """Store paper embedding in vector store"""
    if not vector_store or not VECTOR_STORE_AVAILABLE:
        logger.warning("Vector store not available")
        return False

    try:
        # Combine title and abstract for better representation
        text = f"{title}\n\n{abstract}"
        embedding = await generate_embedding(text)

        if embedding is not None:
            vector_store.add(
                ids=[paper_id],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
            logger.info(f"Stored embedding for paper: {paper_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Vector store failed: {e}")
        return False


async def semantic_search(query: str, n_results: int = 10) -> List[Dict]:
    """Semantic search using vector store"""
    if not vector_store or not VECTOR_STORE_AVAILABLE or not EMBEDDINGS_AVAILABLE:
        logger.warning("Semantic search not available")
        return []

    try:
        query_embedding = await generate_embedding(query)
        if query_embedding is None:
            return []

        results = vector_store.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )

        # Return metadata from results
        return results.get('metadatas', [[]])[0] if results else []
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


async def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts"""
    if not EMBEDDINGS_AVAILABLE:
        # Fallback to simple keyword overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / max(len(words1), len(words2))

    try:
        emb1 = await generate_embedding(text1)
        emb2 = await generate_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Compute cosine similarity
        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]

        return float(similarity)
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        return 0.0


async def find_similar_papers(
    paper_id: str,
    title: str,
    abstract: str,
    n_results: int = 5
) -> List[Dict]:
    """Find papers similar to the given paper"""
    if not VECTOR_STORE_AVAILABLE:
        return []

    try:
        # Search using title and abstract
        query = f"{title} {abstract[:500]}"
        results = await semantic_search(query, n_results=n_results + 1)

        # Filter out the query paper itself
        return [r for r in results if r.get("paper_id") != paper_id][:n_results]
    except Exception as e:
        logger.error(f"Similar papers search failed: {e}")
        return []


async def rag_query(
    question: str,
    context_papers: List[Dict],
    max_context_length: int = 2000
) -> str:
    """
    Retrieval-Augmented Generation for Q&A

    Args:
        question: User's question
        context_papers: List of relevant papers
        max_context_length: Maximum context length

    Returns:
        Context string to pass to LLM
    """
    from app.services.ai_service import gemini_generate_text

    # Build context from papers
    context_parts = []
    current_length = 0

    for paper in context_papers:
        paper_context = f"""
Title: {paper.get('title', 'Unknown')}
Authors: {paper.get('authors', 'Unknown')}
Abstract: {paper.get('abstract', '')[:500]}
"""
        part_length = len(paper_context)

        if current_length + part_length > max_context_length:
            break

        context_parts.append(paper_context)
        current_length += part_length

    context = "\n---\n".join(context_parts)

    # Generate answer using RAG
    prompt = f"""Based on the following research papers, answer this question:

Question: {question}

Context from research papers:
{context}

Provide a concise, evidence-based answer citing the relevant papers."""

    answer = await gemini_generate_text(prompt)
    return answer or "Unable to generate answer from the provided context."


__all__ = [
    "generate_embedding",
    "store_paper_embedding",
    "semantic_search",
    "compute_similarity",
    "find_similar_papers",
    "rag_query",
    "EMBEDDINGS_AVAILABLE",
    "VECTOR_STORE_AVAILABLE"
]
