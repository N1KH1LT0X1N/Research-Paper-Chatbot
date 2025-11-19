"""
AI service - Google Gemini integration for text generation and summarization
"""

import re
from typing import Optional, Dict
from app.core.config import settings, logger

# Import Gemini with fallback
try:
    import google.generativeai as genai
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.warning("Google Generative AI not available")


async def gemini_generate_text(prompt: str, temperature: float = None) -> Optional[str]:
    """Generate text using Gemini"""
    if not GEMINI_AVAILABLE:
        logger.warning("Gemini not available")
        return None

    if temperature is None:
        temperature = settings.TEMPERATURE

    try:
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return getattr(resp, "text", "").strip() or None
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None


async def summarize_paper(
    title: str,
    abstract: str,
    full_text: Optional[str] = None,
    sections: Optional[Dict] = None,
    detailed: bool = False
) -> str:
    """Generate structured summary of a research paper"""

    # Determine source material
    if full_text and detailed:
        source = f"Title: {title}\n\nFull text (excerpt):\n{full_text[:3000]}"
    elif sections and detailed:
        source = f"Title: {title}\n\nSections:\n"
        for name, content in sections.items():
            source += f"\n{name}:\n{content[:500]}\n"
    else:
        source = f"Title: {title}\n\nAbstract: {abstract}"

    style = "Write a concise, structured summary with these sections: Introduction, Methodology, Results, Conclusions."
    if detailed:
        style += " Provide more detail (5-7 sentences per section)."
    else:
        style += " Keep it brief (3-4 sentences per section)."

    prompt = f"""{style}

{source}

Use markdown headings: ## Introduction, ## Methodology, ## Results, ## Conclusions"""

    ai_resp = await gemini_generate_text(prompt)

    if ai_resp:
        return ai_resp

    # Fallback summary when AI is not available
    intro = abstract[:400] if abstract else "Not available."
    return f"""## Introduction
{intro}

## Methodology
Method details not available from abstract.

## Results
Results not explicitly stated in abstract.

## Conclusions
Conclusions inferred from context."""


def parse_structured_sections(text: str) -> Dict[str, str]:
    """Parse markdown sections from summary text"""
    sections = {
        "Introduction": "",
        "Methodology": "",
        "Results": "",
        "Conclusions": ""
    }
    current = None

    for line in text.splitlines():
        m = re.match(
            r"^##\s*(Introduction|Methodology|Methods?|Results?|Conclusions?)\s*$",
            line.strip(),
            re.I
        )
        if m:
            name = m.group(1).capitalize()
            if name.lower().startswith("method"):
                name = "Methodology"
            elif name.lower().startswith("conclusion"):
                name = "Conclusions"
            elif name.lower().startswith("result"):
                name = "Results"
            current = name
            continue

        if current and current in sections:
            sections[current] += (line + "\n")

    # Trim whitespace
    for k in sections:
        sections[k] = sections[k].strip()

    return sections


def compact_summary(
    title: str,
    year: Optional[int],
    authors: str,
    url: str,
    abstract: str,
    max_length: int = 500
) -> str:
    """Create a compact, WhatsApp-friendly summary"""

    # Truncate abstract
    abstract_short = abstract[:max_length] + "..." if len(abstract) > max_length else abstract

    # Build summary
    year_str = f" ({year})" if year else ""
    summary = f"""*{title}*{year_str}

👥 {authors[:100]}

📄 {abstract_short}

🔗 {url}"""

    return summary


async def explain_concept(concept: str, context: Optional[str] = None) -> Optional[str]:
    """Explain a scientific concept in simple terms"""

    prompt = f"Explain the concept of '{concept}' in 2-3 simple sentences suitable for a researcher."
    if context:
        prompt += f"\n\nContext: {context[:500]}"

    return await gemini_generate_text(prompt, temperature=0.3)


async def generate_keywords(title: str, abstract: str) -> list[str]:
    """Extract keywords from paper title and abstract"""

    prompt = f"""Extract 5-7 key technical keywords from this research paper.
Return only the keywords, comma-separated.

Title: {title}
Abstract: {abstract[:500]}"""

    response = await gemini_generate_text(prompt, temperature=0.2)

    if response:
        keywords = [k.strip() for k in response.split(",")]
        return keywords[:7]

    # Fallback: simple extraction
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', title + " " + abstract)
    return list(set(words))[:7]


__all__ = [
    "gemini_generate_text",
    "summarize_paper",
    "parse_structured_sections",
    "compact_summary",
    "explain_concept",
    "generate_keywords",
    "GEMINI_AVAILABLE"
]
