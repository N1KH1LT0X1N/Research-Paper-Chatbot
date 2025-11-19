"""
Citation Export Module
Support for BibTeX, RIS, CSV, and other formats
"""

import re
from typing import List, Dict
from datetime import datetime


class CitationExporter:
    """Export citations in various formats"""

    @staticmethod
    def to_bibtex(papers: List[Dict]) -> str:
        """Export papers as BibTeX"""
        entries = []

        for paper in papers:
            # Generate citation key
            first_author = paper.get('authors', 'Unknown').split(',')[0].split()[-1]
            year = paper.get('year', '????')
            title_word = re.sub(r'[^a-zA-Z]', '', paper.get('title', '').split()[0])
            key = f"{first_author}{year}{title_word}".lower()

            # Determine entry type
            venue = paper.get('venue', '')
            if 'arxiv' in venue.lower():
                entry_type = 'article'
                journal = 'arXiv preprint'
            elif 'conference' in venue.lower() or 'proceedings' in venue.lower():
                entry_type = 'inproceedings'
                journal = venue
            else:
                entry_type = 'article'
                journal = venue or 'Unknown'

            # Build entry
            entry = f"""@{entry_type}{{{key},
  title={{{paper.get('title', 'Unknown')}}},
  author={{{paper.get('authors', 'Unknown')}}},
  year={{{year}}},
  journal={{{journal}}},
  url={{{paper.get('url', '')}}}"""

            # Add optional fields
            if paper.get('abstract'):
                # Truncate abstract for BibTeX
                abstract = paper['abstract'][:500] + '...' if len(paper['abstract']) > 500 else paper['abstract']
                entry += f",\n  abstract={{{abstract}}}"

            entry += "\n}"
            entries.append(entry)

        return "\n\n".join(entries)

    @staticmethod
    def to_ris(papers: List[Dict]) -> str:
        """Export papers as RIS format (for EndNote, Zotero, Mendeley)"""
        entries = []

        for paper in papers:
            entry_lines = []

            # Type of reference
            venue = paper.get('venue', '')
            if 'arxiv' in venue.lower():
                entry_lines.append("TY  - JOUR")  # Journal article
            elif 'conference' in venue.lower():
                entry_lines.append("TY  - CONF")  # Conference paper
            else:
                entry_lines.append("TY  - JOUR")

            # Title
            entry_lines.append(f"T1  - {paper.get('title', 'Unknown')}")

            # Authors (one line per author)
            authors = paper.get('authors', 'Unknown').split(',')
            for author in authors:
                entry_lines.append(f"AU  - {author.strip()}")

            # Year
            if paper.get('year'):
                entry_lines.append(f"PY  - {paper['year']}")

            # Journal/Venue
            if paper.get('venue'):
                entry_lines.append(f"JO  - {paper['venue']}")

            # Abstract
            if paper.get('abstract'):
                entry_lines.append(f"AB  - {paper['abstract']}")

            # URL
            if paper.get('url'):
                entry_lines.append(f"UR  - {paper['url']}")

            # End of record
            entry_lines.append("ER  -")

            entries.append("\n".join(entry_lines))

        return "\n\n".join(entries)

    @staticmethod
    def to_csv(papers: List[Dict]) -> str:
        """Export papers as CSV"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Title',
            'Authors',
            'Year',
            'Venue',
            'Citations',
            'URL',
            'Abstract'
        ])

        # Data
        for paper in papers:
            writer.writerow([
                paper.get('title', ''),
                paper.get('authors', ''),
                paper.get('year', ''),
                paper.get('venue', ''),
                paper.get('citationCount', 0),
                paper.get('url', ''),
                paper.get('abstract', '')[:200] + '...' if len(paper.get('abstract', '')) > 200 else paper.get('abstract', '')
            ])

        return output.getvalue()

    @staticmethod
    def to_markdown(papers: List[Dict]) -> str:
        """Export papers as Markdown list"""
        lines = ["# Reading List\n"]
        lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        for i, paper in enumerate(papers, 1):
            lines.append(f"## {i}. {paper.get('title', 'Unknown')}")
            lines.append(f"**Authors:** {paper.get('authors', 'Unknown')}")
            lines.append(f"**Year:** {paper.get('year', '?')}")

            if paper.get('venue'):
                lines.append(f"**Venue:** {paper['venue']}")

            if paper.get('citationCount'):
                lines.append(f"**Citations:** {paper['citationCount']}")

            if paper.get('url'):
                lines.append(f"**URL:** [{paper['url']}]({paper['url']})")

            if paper.get('abstract'):
                lines.append(f"\n**Abstract:** {paper['abstract'][:300]}...\n")
            else:
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_plain_text(papers: List[Dict]) -> str:
        """Export papers as plain text"""
        lines = []

        for i, paper in enumerate(papers, 1):
            lines.append(f"{i}. {paper.get('title', 'Unknown')}")
            lines.append(f"   {paper.get('authors', 'Unknown')} ({paper.get('year', '?')})")

            if paper.get('url'):
                lines.append(f"   {paper['url']}")

            lines.append("")

        return "\n".join(lines)


class CitationGraphNavigator:
    """Navigate citation networks"""

    def __init__(self, http_client):
        self.http_client = http_client

    async def get_citations(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Get papers that cite this paper"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations"
        params = {
            "fields": "citingPaper.paperId,citingPaper.title,citingPaper.authors,citingPaper.year,citingPaper.citationCount,contexts,intents",
            "limit": limit
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            citations = []
            for item in data.get('data', []):
                citing_paper = item.get('citingPaper', {})
                citations.append({
                    'paperId': citing_paper.get('paperId'),
                    'title': citing_paper.get('title'),
                    'authors': ', '.join([a.get('name', '') for a in citing_paper.get('authors', [])]),
                    'year': citing_paper.get('year'),
                    'citationCount': citing_paper.get('citationCount', 0),
                    'contexts': item.get('contexts', []),
                    'intents': item.get('intents', [])
                })

            return citations

        except Exception as e:
            print(f"Failed to fetch citations: {e}")
            return []

    async def get_references(self, paper_id: str, limit: int = 10) -> List[Dict]:
        """Get papers referenced by this paper"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        params = {
            "fields": "citedPaper.paperId,citedPaper.title,citedPaper.authors,citedPaper.year,citedPaper.citationCount,contexts,intents",
            "limit": limit
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            references = []
            for item in data.get('data', []):
                cited_paper = item.get('citedPaper', {})
                references.append({
                    'paperId': cited_paper.get('paperId'),
                    'title': cited_paper.get('title'),
                    'authors': ', '.join([a.get('name', '') for a in cited_paper.get('authors', [])]),
                    'year': cited_paper.get('year'),
                    'citationCount': cited_paper.get('citationCount', 0),
                    'contexts': item.get('contexts', []),
                    'intents': item.get('intents', [])
                })

            return references

        except Exception as e:
            print(f"Failed to fetch references: {e}")
            return []

    async def get_related_papers(self, paper_id: str, limit: int = 5) -> List[Dict]:
        """Get papers related to this one (recommendations from Semantic Scholar)"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        params = {
            "fields": "title,authors,year,abstract,citationCount,references,citations"
        }

        try:
            # First get the paper details
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            paper_data = response.json()

            # Use references and citations to find related work
            # Combine most cited references and most cited papers that cite this
            references = await self.get_references(paper_id, limit=limit*2)
            citations = await self.get_citations(paper_id, limit=limit*2)

            # Sort by citation count and combine
            all_related = references + citations
            all_related.sort(key=lambda x: x.get('citationCount', 0), reverse=True)

            # Deduplicate
            seen = set()
            unique_related = []
            for paper in all_related:
                if paper['paperId'] not in seen:
                    seen.add(paper['paperId'])
                    unique_related.append(paper)

            return unique_related[:limit]

        except Exception as e:
            print(f"Failed to fetch related papers: {e}")
            return []

    def format_citation_context(self, citation: Dict) -> str:
        """Format how and why a paper was cited"""
        lines = []

        lines.append(f"📄 *{citation['title']}* ({citation.get('year', '?')})")
        lines.append(f"👤 {citation['authors'][:60]}...")

        # Intent (why cited)
        intents = citation.get('intents', [])
        if intents:
            intent_str = ', '.join(intents)
            lines.append(f"🎯 Intent: {intent_str}")

        # Context (where cited)
        contexts = citation.get('contexts', [])
        if contexts:
            lines.append(f"\n💬 Context:")
            for ctx in contexts[:2]:  # Show first 2 contexts
                # Truncate long contexts
                ctx_text = ctx[:200] + '...' if len(ctx) > 200 else ctx
                lines.append(f"   \"{ctx_text}\"")

        return "\n".join(lines)


# Example usage functions

async def export_reading_list_to_file(papers: List[Dict], format: str = 'bibtex') -> str:
    """
    Export reading list to specified format
    Returns: formatted string ready to save/send
    """
    exporter = CitationExporter()

    if format.lower() == 'bibtex':
        return exporter.to_bibtex(papers)
    elif format.lower() == 'ris':
        return exporter.to_ris(papers)
    elif format.lower() == 'csv':
        return exporter.to_csv(papers)
    elif format.lower() == 'markdown':
        return exporter.to_markdown(papers)
    elif format.lower() == 'txt':
        return exporter.to_plain_text(papers)
    else:
        raise ValueError(f"Unsupported format: {format}")
