"""
Services package - Business logic modules
"""

from app.services.paper_search import (
    search_papers_semantic_scholar,
    search_papers_arxiv,
    search_papers_combined,
    get_paper_by_doi,
    get_paper_by_arxiv_id
)

from app.services.pdf_processor import (
    download_pdf_from_arxiv,
    download_pdf_from_url,
    extract_pdf_text_async,
    parse_paper_sections,
    extract_figures_from_pdf,
    PDF_AVAILABLE
)

from app.services.ai_service import (
    gemini_generate_text,
    summarize_paper,
    parse_structured_sections,
    compact_summary,
    explain_concept,
    generate_keywords,
    GEMINI_AVAILABLE
)

from app.services.vector_search import (
    generate_embedding,
    store_paper_embedding,
    semantic_search,
    compute_similarity,
    find_similar_papers,
    rag_query,
    EMBEDDINGS_AVAILABLE,
    VECTOR_STORE_AVAILABLE
)

from app.services.qna_service import (
    generate_qna_items,
    evaluate_answer_semantic,
    grade_with_llm,
    generate_feedback,
    calculate_answer_score_simple
)

from app.services.spaced_repetition import (
    schedule_review,
    get_due_reviews,
    get_review_stats,
    get_next_review_time,
    calculate_retention_score
)

from app.services.analytics import (
    generate_user_stats,
    calculate_streak,
    recommend_papers,
    get_activity_summary,
    award_achievement,
    check_and_award_achievements
)

__all__ = [
    # Paper search
    "search_papers_semantic_scholar",
    "search_papers_arxiv",
    "search_papers_combined",
    "get_paper_by_doi",
    "get_paper_by_arxiv_id",

    # PDF processing
    "download_pdf_from_arxiv",
    "download_pdf_from_url",
    "extract_pdf_text_async",
    "parse_paper_sections",
    "extract_figures_from_pdf",
    "PDF_AVAILABLE",

    # AI service
    "gemini_generate_text",
    "summarize_paper",
    "parse_structured_sections",
    "compact_summary",
    "explain_concept",
    "generate_keywords",
    "GEMINI_AVAILABLE",

    # Vector search
    "generate_embedding",
    "store_paper_embedding",
    "semantic_search",
    "compute_similarity",
    "find_similar_papers",
    "rag_query",
    "EMBEDDINGS_AVAILABLE",
    "VECTOR_STORE_AVAILABLE",

    # Q&A
    "generate_qna_items",
    "evaluate_answer_semantic",
    "grade_with_llm",
    "generate_feedback",
    "calculate_answer_score_simple",

    # Spaced repetition
    "schedule_review",
    "get_due_reviews",
    "get_review_stats",
    "get_next_review_time",
    "calculate_retention_score",

    # Analytics
    "generate_user_stats",
    "calculate_streak",
    "recommend_papers",
    "get_activity_summary",
    "award_achievement",
    "check_and_award_achievements",
]