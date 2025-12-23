"""
Database models for Research Paper Chatbot
"""

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey, Index
from datetime import datetime

Base = declarative_base()


class Session(Base):
    """User session model"""
    __tablename__ = "sessions"

    user_id = Column(String, primary_key=True)
    mode = Column(String, default="browsing")  # browsing | qna | review
    selected_paper_id = Column(String)
    selected_paper_title = Column(String)
    selected_paper_abstract = Column(Text)
    qna_active = Column(Boolean, default=False)
    qna_index = Column(Integer, default=0)
    qna_questions = Column(JSON)
    score = Column(Integer, default=0)
    last_results = Column(JSON)
    current_list = Column(String)
    current_group = Column(String)
    difficulty_preference = Column(String, default="medium")
    voice_enabled = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)


class Paper(Base):
    """Research paper model"""
    __tablename__ = "papers"

    paper_id = Column(String, primary_key=True)
    title = Column(String)
    authors = Column(String)
    year = Column(Integer)
    venue = Column(String)
    url = Column(String)
    abstract = Column(Text)
    full_text = Column(Text)
    pdf_path = Column(String)
    citation_count = Column(Integer, default=0)
    paper_metadata = Column(JSON)  # Renamed from metadata (reserved keyword)
    sections = Column(JSON)
    figures = Column(JSON)
    tables = Column(JSON)
    references = Column(JSON)
    embeddings = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_paper_title', 'title'),)


class UserHistory(Base):
    """User activity history"""
    __tablename__ = "user_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    action = Column(String)  # searched | read | qna_completed | added_to_list
    score = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_user_history', 'user_id', 'timestamp'),)


class ReadingList(Base):
    """User reading lists"""
    __tablename__ = "reading_lists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    list_name = Column(String, default="default")
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    status = Column(String, default="to_read")  # to_read | reading | completed
    notes = Column(Text)
    score = Column(Integer)
    added_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    __table_args__ = (Index('idx_reading_list', 'user_id', 'list_name'),)


class StudyGroup(Base):
    """Study groups"""
    __tablename__ = "study_groups"

    id = Column(Integer, primary_key=True, autoincrement=True)
    group_code = Column(String, unique=True)
    name = Column(String)
    created_by = Column(String)
    members = Column(JSON)
    papers = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class Achievement(Base):
    """User achievements"""
    __tablename__ = "achievements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    achievement_key = Column(String)
    unlocked_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index('idx_achievements', 'user_id'),)


class ReviewSchedule(Base):
    """Spaced repetition schedule"""
    __tablename__ = "review_schedule"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    paper_id = Column(String, ForeignKey("papers.paper_id"))
    question_id = Column(String)
    easiness = Column(Float, default=2.5)
    interval = Column(Integer, default=1)
    repetitions = Column(Integer, default=0)
    next_review = Column(DateTime)
    last_score = Column(Float)

    __table_args__ = (Index('idx_review_schedule', 'user_id', 'next_review'),)


class ChatLog(Base):
    """Chat message logs"""
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String)
    role = Column(String)  # user | bot
    message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


__all__ = [
    "Base",
    "Session",
    "Paper",
    "UserHistory",
    "ReadingList",
    "StudyGroup",
    "Achievement",
    "ReviewSchedule",
    "ChatLog"
]
