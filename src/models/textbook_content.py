from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class TextbookContent(BaseModel):
    """
    Structured educational material from the 6 textbook modules
    """
    id: UUID
    module_id: str
    chapter_id: str
    section_id: str
    content_type: str  # Enum: text, code, diagram_description, etc.
    content: str
    hierarchy_path: str
    embedding: Optional[list] = None  # Vector embedding (generated using GEMINI embeddings)
    metadata: Optional[dict] = None  # Additional metadata (word count, reading level, etc.)

    class Config:
        from_attributes = True


class TextbookContentCreate(BaseModel):
    module_id: str
    chapter_id: str
    section_id: str
    content_type: str
    content: str
    hierarchy_path: str
    embedding: Optional[list] = None
    metadata: Optional[dict] = None


class TextbookContentUpdate(BaseModel):
    content: Optional[str] = None
    metadata: Optional[dict] = None


# For database modeling using SQLAlchemy
from sqlalchemy import Column, DateTime, String, Text, JSON, UUID, Enum
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, ARRAY
from sqlalchemy.sql import func

Base = declarative_base()


class TextbookContentDB(Base):
    __tablename__ = "textbook_content"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    module_id = Column(String, nullable=False)
    chapter_id = Column(String, nullable=False)
    section_id = Column(String, nullable=False)
    content_type = Column(String, nullable=False)  # text, code, diagram_description, etc.
    content = Column(Text, nullable=False)
    hierarchy_path = Column(String, nullable=False)
    # In a real implementation, the embedding would be stored in Qdrant, not in Postgres
    # For now, we represent it as a JSON array of floats
    embedding = Column(JSON)  # Vector embedding generated using GEMINI embeddings
    content_metadata = Column(JSON)  # Additional metadata (word count, reading level, etc.)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())