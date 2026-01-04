from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class ChatSession(BaseModel):
    """
    Represents a single conversation session between student and chatbot
    """
    id: UUID
    student_id: UUID
    created_at: datetime
    updated_at: datetime
    current_module_context: Optional[str] = None
    is_active: bool = True
    session_metadata: Optional[dict] = None
    # Conversation context tracking
    conversation_context: Optional[dict] = None  # Stores current conversation topic, user intent, etc.
    last_interaction_at: Optional[datetime] = None  # Timestamp of last interaction
    conversation_depth: int = 0  # Number of exchanges in the conversation
    active_topic: Optional[str] = None  # Current topic being discussed

    class Config:
        from_attributes = True


class ChatSessionCreate(BaseModel):
    student_id: UUID
    current_module_context: Optional[str] = None
    session_metadata: Optional[dict] = None
    # Conversation context tracking
    conversation_context: Optional[dict] = None
    active_topic: Optional[str] = None


class ChatSessionUpdate(BaseModel):
    current_module_context: Optional[str] = None
    is_active: Optional[bool] = None
    session_metadata: Optional[dict] = None
    # Conversation context tracking
    conversation_context: Optional[dict] = None
    active_topic: Optional[str] = None
    last_interaction_at: Optional[datetime] = None


# For database modeling using SQLAlchemy
from sqlalchemy import Column, DateTime, String, Text, JSON, Boolean, UUID, ForeignKey, Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.sql import func

Base = declarative_base()


class ChatSessionDB(Base):
    __tablename__ = "chat_sessions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    student_id = Column(PostgresUUID(as_uuid=True), ForeignKey("students.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    current_module_context = Column(String)
    is_active = Column(Boolean, default=True)
    session_metadata = Column(JSON)
    # Conversation context tracking
    conversation_context = Column(JSON)  # Stores current conversation topic, user intent, etc.
    last_interaction_at = Column(DateTime)  # Timestamp of last interaction
    conversation_depth = Column(Integer, default=0)  # Number of exchanges in the conversation
    active_topic = Column(String)  # Current topic being discussed