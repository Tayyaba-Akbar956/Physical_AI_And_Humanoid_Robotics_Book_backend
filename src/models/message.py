from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
from uuid import UUID, uuid4


class Message(BaseModel):
    """
    Individual message within a chat session
    """
    id: UUID
    session_id: UUID
    sender_type: str  # Enum: student, ai_agent
    content: str
    timestamp: datetime
    message_metadata: Optional[dict] = None
    citations: Optional[List[dict]] = None  # References to textbook modules/chapters used in response
    selected_text_ref: Optional[UUID] = None  # Reference to selected text if this is a text selection query
    # Conversation context tracking
    conversation_turn: Optional[int] = None  # Position in the conversation (0, 1, 2, ...)
    context_before: Optional[List[dict]] = None  # Previous messages in the conversation
    context_after: Optional[List[dict]] = None  # Following messages in the conversation
    parent_message_id: Optional[UUID] = None  # ID of parent message this is responding to
    topic_anchored: Optional[str] = None  # Topic this message is anchored to
    follow_up_to: Optional[UUID] = None  # ID of message this is a follow-up to

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    session_id: UUID
    sender_type: str
    content: str
    message_metadata: Optional[dict] = None
    citations: Optional[List[dict]] = None
    selected_text_ref: Optional[UUID] = None
    # Conversation context tracking
    conversation_turn: Optional[int] = None
    parent_message_id: Optional[UUID] = None
    topic_anchored: Optional[str] = None
    follow_up_to: Optional[UUID] = None


class MessageUpdate(BaseModel):
    content: Optional[str] = None
    message_metadata: Optional[dict] = None
    citations: Optional[List[dict]] = None
    # Conversation context tracking
    topic_anchored: Optional[str] = None


# For database modeling using SQLAlchemy
from sqlalchemy import Column, DateTime, String, Text, JSON, UUID, ForeignKey, Integer
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.sql import func

Base = declarative_base()


class MessageDB(Base):
    __tablename__ = "messages"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PostgresUUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    sender_type = Column(String, nullable=False)  # student, ai_agent
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now())
    message_metadata = Column(JSON)
    citations = Column(JSON)  # References to textbook modules/chapters used in response
    selected_text_ref = Column(PostgresUUID(as_uuid=True), ForeignKey("selected_texts.id"))  # Reference to selected text if this is a text selection query
    # Conversation context tracking
    conversation_turn = Column(Integer)  # Position in the conversation (0, 1, 2, ...)
    context_before = Column(JSON)  # Previous messages in the conversation
    context_after = Column(JSON)  # Following messages in the conversation
    parent_message_id = Column(PostgresUUID(as_uuid=True), ForeignKey("messages.id"))  # ID of parent message this is responding to
    topic_anchored = Column(String)  # Topic this message is anchored to
    follow_up_to = Column(PostgresUUID(as_uuid=True), ForeignKey("messages.id"))  # ID of message this is a follow-up to