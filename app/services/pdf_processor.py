"""
PDF processor service - Download and extract content from PDFs
"""

import os
import re
import tempfile
from typing import Dict, Any, Optional
from app.core.config import logger, settings

# Import PDF libraries with fallback
try:
    import pdfplumber
    import PyPDF2
    from PIL import Image
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("PDF processing libraries not available")

# Import HTTP client from paper_search
from app.services.paper_search import http_client


async def download_pdf_from_arxiv(arxiv_id: str) -> Optional[bytes]:
    """Download PDF from arXiv"""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    try:
        response = await http_client.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"arXiv PDF download failed for {arxiv_id}: {e}")
        return None


async def download_pdf_from_url(url: str) -> Optional[bytes]:
    """Download PDF from any URL"""
    try:
        response = await http_client.get(url, follow_redirects=True, timeout=60.0)
        response.raise_for_status()

        # Verify it's a PDF
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            logger.warning(f"URL does not return PDF: {content_type}")
            return None

        return response.content
    except Exception as e:
        logger.error(f"PDF download failed from {url}: {e}")
        return None


def extract_pdf_text(pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract text and metadata from PDF (synchronous)"""
    if not PDF_AVAILABLE:
        logger.warning("PDF libraries not available")
        return {"full_text": "", "sections": {}, "page_count": 0}

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        full_text = ""
        page_count = 0

        # Try pdfplumber first (better text extraction)
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"

                page_count = len(pdf.pages)
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            # Fallback to PyPDF2
            try:
                with open(tmp_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        full_text += page.extract_text() + "\n"
                    page_count = len(pdf_reader.pages)
            except Exception as e2:
                logger.error(f"PyPDF2 also failed: {e2}")

        # Parse sections
        sections = parse_paper_sections(full_text)

        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

        return {
            "full_text": full_text,
            "sections": sections,
            "page_count": page_count
        }
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return {"full_text": "", "sections": {}, "page_count": 0}


async def extract_pdf_text_async(pdf_bytes: bytes) -> Dict[str, Any]:
    """Async wrapper for PDF text extraction"""
    import asyncio
    # Run blocking PDF extraction in thread pool
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_pdf_text, pdf_bytes)


def parse_paper_sections(text: str) -> Dict[str, str]:
    """Parse paper into sections using heuristics"""
    sections = {}

    # Common section patterns
    patterns = {
        "Abstract": r'\n\s*(?:ABSTRACT|Abstract)\s*\n(.*?)(?=\n\s*(?:\d+\.?\s*)?(?:INTRODUCTION|Introduction|I\.|1\.))',
        "Introduction": r'\n\s*(?:1\.?\s*)?(?:INTRODUCTION|Introduction)\s*\n(.*?)(?=\n\s*(?:2\.?|II\.)\s*(?:RELATED|Background|METHOD))',
        "Methods": r'\n\s*(?:\d+\.?\s*)?(?:METHODS?|Methodology|APPROACH|Approach)\s*\n(.*?)(?=\n\s*(?:\d+\.?)\s*(?:RESULTS?|EXPERIMENTS?))',
        "Results": r'\n\s*(?:\d+\.?\s*)?(?:RESULTS?|EXPERIMENTS?|Evaluation)\s*\n(.*?)(?=\n\s*(?:\d+\.?)\s*(?:DISCUSSION|CONCLUSION))',
        "Conclusion": r'\n\s*(?:\d+\.?\s*)?(?:CONCLUSION|Conclusions?|DISCUSSION)\s*\n(.*?)(?=\n\s*(?:REFERENCES?|Bibliography|ACKNOWLEDGE))',
    }

    for section_name, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content)
            sections[section_name] = content[:2000]  # Limit length

    return sections


def extract_figures_from_pdf(pdf_bytes: bytes) -> list[Dict[str, Any]]:
    """Extract figures/images from PDF"""
    if not PDF_AVAILABLE:
        return []

    figures = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        with pdfplumber.open(tmp_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract images
                if hasattr(page, 'images'):
                    for img_idx, img in enumerate(page.images):
                        figures.append({
                            "page": page_num + 1,
                            "index": img_idx,
                            "bbox": (img.get("x0"), img.get("top"),
                                   img.get("x1"), img.get("bottom")),
                            "type": "image"
                        })

        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

        return figures
    except Exception as e:
        logger.error(f"Figure extraction failed: {e}")
        return []


__all__ = [
    "download_pdf_from_arxiv",
    "download_pdf_from_url",
    "extract_pdf_text",
    "extract_pdf_text_async",
    "parse_paper_sections",
    "extract_figures_from_pdf",
    "PDF_AVAILABLE"
]
