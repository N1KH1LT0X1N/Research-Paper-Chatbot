"""
Learning Path Generator
Creates structured learning paths for topics
"""

from typing import List, Dict, Optional
import asyncio


class LearningPathGenerator:
    """Generate personalized learning paths"""

    def __init__(self, http_client, gemini_client=None):
        self.http_client = http_client
        self.gemini = gemini_client

    async def create_learning_path(
        self,
        topic: str,
        user_level: str = "beginner",
        num_papers: int = 8
    ) -> Dict:
        """
        Create a structured learning path

        Args:
            topic: Topic to learn (e.g., "deep learning", "transformers")
            user_level: beginner | intermediate | advanced
            num_papers: Number of papers in the path

        Returns:
            Dict with learning path structure
        """

        # Step 1: Find foundational papers
        foundational = await self._find_foundational_papers(topic)

        # Step 2: Find recent survey/review papers
        surveys = await self._find_survey_papers(topic)

        # Step 3: Find recent influential papers
        recent = await self._find_recent_papers(topic)

        # Step 4: Order by difficulty and dependencies
        ordered_papers = await self._order_papers_by_difficulty(
            foundational, surveys, recent, user_level, num_papers
        )

        # Step 5: Generate explanations for each step
        path = []
        for i, paper in enumerate(ordered_papers):
            explanation = await self._explain_paper_in_path(paper, topic, i, user_level)

            path.append({
                'step': i + 1,
                'paper_id': paper.get('paperId'),
                'title': paper.get('title'),
                'authors': paper.get('authors'),
                'year': paper.get('year'),
                'url': paper.get('url'),
                'abstract': paper.get('abstract', '')[:300],
                'difficulty': self._estimate_difficulty(paper, user_level),
                'estimated_time': self._estimate_reading_time(paper),
                'reason': explanation,
                'prerequisites': self._get_prerequisites(i, path)
            })

        return {
            'topic': topic,
            'level': user_level,
            'total_papers': len(path),
            'estimated_hours': sum(p['estimated_time'] for p in path),
            'path': path
        }

    async def _find_foundational_papers(self, topic: str, limit: int = 5) -> List[Dict]:
        """Find seminal/foundational papers"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "fields": "paperId,title,authors,year,citationCount,influentialCitationCount,abstract,url",
            "limit": 30,  # Get more to filter
            "sort": "citationCount:desc"
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = data.get('data', [])

            # Filter to older, highly cited papers (foundational)
            foundational = [
                p for p in papers
                if p.get('year') and p.get('year') < 2018  # Older papers
                and p.get('citationCount', 0) > 500  # Highly cited
            ]

            return foundational[:limit]

        except Exception as e:
            print(f"Failed to find foundational papers: {e}")
            return []

    async def _find_survey_papers(self, topic: str, limit: int = 2) -> List[Dict]:
        """Find survey/review papers"""
        # Search with 'survey' or 'review' keywords
        survey_query = f"{topic} survey review"

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": survey_query,
            "fields": "paperId,title,authors,year,citationCount,abstract,url",
            "limit": 10
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = data.get('data', [])

            # Filter papers with 'survey' or 'review' in title
            surveys = [
                p for p in papers
                if 'survey' in p.get('title', '').lower()
                or 'review' in p.get('title', '').lower()
            ]

            return surveys[:limit]

        except Exception as e:
            print(f"Failed to find survey papers: {e}")
            return []

    async def _find_recent_papers(self, topic: str, limit: int = 5) -> List[Dict]:
        """Find recent influential papers"""
        from datetime import datetime
        current_year = datetime.now().year

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "fields": "paperId,title,authors,year,citationCount,influentialCitationCount,abstract,url",
            "limit": 30,
            "year": f"{current_year-3}-{current_year}",  # Last 3 years
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            papers = data.get('data', [])

            # Sort by influential citations (quality over quantity)
            papers.sort(
                key=lambda p: p.get('influentialCitationCount', 0),
                reverse=True
            )

            return papers[:limit]

        except Exception as e:
            print(f"Failed to find recent papers: {e}")
            return []

    async def _order_papers_by_difficulty(
        self,
        foundational: List[Dict],
        surveys: List[Dict],
        recent: List[Dict],
        user_level: str,
        num_papers: int
    ) -> List[Dict]:
        """Order papers from easy to hard based on user level"""

        all_papers = []

        if user_level == "beginner":
            # Start with surveys, then foundational, then recent
            all_papers.extend(surveys[:2])
            all_papers.extend(foundational[:3])
            all_papers.extend(recent[:3])

        elif user_level == "intermediate":
            # Start with one survey, mix foundational and recent
            all_papers.extend(surveys[:1])
            all_papers.extend(foundational[:2])
            all_papers.extend(recent[:5])

        else:  # advanced
            # Focus on recent work, with foundational context
            all_papers.extend(foundational[:1])
            all_papers.extend(recent[:6])
            all_papers.extend(surveys[:1])

        # Deduplicate
        seen = set()
        unique_papers = []
        for paper in all_papers:
            paper_id = paper.get('paperId')
            if paper_id and paper_id not in seen:
                seen.add(paper_id)
                unique_papers.append(paper)

        return unique_papers[:num_papers]

    async def _explain_paper_in_path(
        self,
        paper: Dict,
        topic: str,
        step: int,
        level: str
    ) -> str:
        """Generate explanation for why this paper is in the path"""

        if self.gemini:
            # Use AI to generate explanation
            context = "first" if step == 0 else "intermediate" if step < 4 else "advanced"

            prompt = f"""Explain in 2-3 sentences why a {level} learner studying "{topic}" should read this paper at the {context} stage:

Paper: {paper.get('title')} ({paper.get('year')})
Abstract: {paper.get('abstract', '')[:300]}

Focus on: what they'll learn, why it's important at this stage, how it builds on previous knowledge."""

            try:
                from async_research_bot import gemini_generate_text
                explanation = await gemini_generate_text(prompt, temperature=0.3)
                if explanation:
                    return explanation
            except:
                pass

        # Fallback explanation
        if step == 0:
            return "Starting point: Introduces core concepts and foundations."
        elif step < 3:
            return "Builds on fundamentals with key theoretical developments."
        elif step < 6:
            return "Explores practical applications and modern approaches."
        else:
            return "Advanced topics and cutting-edge research directions."

    def _estimate_difficulty(self, paper: Dict, user_level: str) -> str:
        """Estimate paper difficulty"""
        year = paper.get('year', 2020)
        citations = paper.get('citationCount', 0)

        # Older, highly cited = foundational (medium difficulty)
        # Recent, many citations = advanced
        # Surveys = easier

        title = paper.get('title', '').lower()

        if 'survey' in title or 'review' in title:
            return 'easy'
        elif year < 2010 and citations > 1000:
            return 'medium'  # Classic paper
        elif year > 2020:
            return 'hard'  # Recent, potentially advanced
        else:
            return 'medium'

    def _estimate_reading_time(self, paper: Dict) -> float:
        """Estimate reading time in hours"""
        # Rough estimates based on paper type
        title = paper.get('title', '').lower()

        if 'survey' in title or 'review' in title:
            return 2.5  # Surveys are long
        else:
            return 1.5  # Average research paper

    def _get_prerequisites(self, step: int, path: List[Dict]) -> List[int]:
        """Get prerequisite steps"""
        if step == 0:
            return []
        elif step == 1:
            return [1]
        else:
            # Last 1-2 papers
            return [step - 1] if step > 1 else []

    def format_learning_path(self, path_data: Dict) -> str:
        """Format learning path for WhatsApp"""
        lines = []

        lines.append(f"🎓 *Learning Path: {path_data['topic'].title()}*")
        lines.append(f"📊 Level: {path_data['level'].title()}")
        lines.append(f"📚 {path_data['total_papers']} papers")
        lines.append(f"⏱ ~{path_data['estimated_hours']:.0f} hours total\n")

        difficulty_icons = {
            'easy': '🟢',
            'medium': '🟡',
            'hard': '🔴'
        }

        for paper in path_data['path'][:5]:  # Show first 5
            icon = difficulty_icons.get(paper['difficulty'], '⚪')

            lines.append(f"\n*Step {paper['step']}: {paper['difficulty'].title()}* {icon}")
            lines.append(f"📄 {paper['title']}")
            lines.append(f"👤 {paper['authors'][:50]}...")
            lines.append(f"⏱ ~{paper['estimated_time']:.1f}h")
            lines.append(f"\n💡 {paper['reason']}")

        if path_data['total_papers'] > 5:
            lines.append(f"\n... and {path_data['total_papers'] - 5} more papers")

        lines.append(f"\nType 'start path' to begin with Step 1!")

        return "\n".join(lines)


class PaperComparator:
    """Compare multiple papers"""

    def __init__(self, gemini_client=None):
        self.gemini = gemini_client

    async def compare_papers(self, papers: List[Dict]) -> str:
        """Generate comparison of 2-5 papers"""

        if len(papers) < 2:
            return "Need at least 2 papers to compare."

        if len(papers) > 5:
            papers = papers[:5]  # Limit to 5

        # Build comparison prompt
        prompt = f"""Compare these {len(papers)} research papers across multiple dimensions:

"""

        for i, paper in enumerate(papers, 1):
            prompt += f"""Paper {i}: {paper.get('title')}
Authors: {paper.get('authors')}
Year: {paper.get('year')}
Abstract: {paper.get('abstract', '')[:500]}

"""

        prompt += """
Generate a structured comparison covering:
1. Main contributions (what's unique about each?)
2. Methodologies (how do approaches differ?)
3. Results & findings (what did each discover?)
4. Strengths & limitations
5. Practical applications

Keep it concise (under 1000 words). Use clear sections and bullet points."""

        if self.gemini:
            try:
                from async_research_bot import gemini_generate_text
                comparison = await gemini_generate_text(prompt, temperature=0.3)
                if comparison:
                    return comparison
            except:
                pass

        # Fallback comparison
        lines = ["*Paper Comparison*\n"]

        for i, paper in enumerate(papers, 1):
            lines.append(f"\n*Paper {i}:* {paper.get('title')}")
            lines.append(f"- Year: {paper.get('year')}")
            lines.append(f"- Citations: {paper.get('citationCount', 0)}")

        lines.append("\nFor detailed comparison, AI generation is required.")

        return "\n".join(lines)

    async def generate_synthesis(self, papers: List[Dict], topic: str) -> str:
        """Generate literature review synthesis"""

        prompt = f"""You are writing a literature review on: {topic}

Synthesize these {len(papers)} papers into a cohesive narrative:

"""

        for i, paper in enumerate(papers, 1):
            prompt += f"{i}. {paper.get('title')} ({paper.get('year')})\n"
            prompt += f"   {paper.get('abstract', '')[:400]}\n\n"

        prompt += """
Write a synthesis covering:
1. Overview of the field
2. Key themes and trends
3. Evolution of ideas (chronologically)
4. Consensus and debates
5. Research gaps
6. Future directions

Write in academic style, 500-700 words."""

        if self.gemini:
            try:
                from async_research_bot import gemini_generate_text
                synthesis = await gemini_generate_text(prompt, temperature=0.4)
                if synthesis:
                    return synthesis
            except:
                pass

        return "Literature synthesis requires AI generation."
