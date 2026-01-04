from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class SelectedText(BaseModel):
    """
    Portion of textbook content highlighted by the student
    """
    id: UUID
    content: str  # The actual selected text (minimum 20 characters)
    module_id: str  # Module where text was selected
    chapter_id: str  # Chapter where text was selected
    section_id: str  # Section where text was selected
    hierarchy_path: str  # Full path in textbook hierarchy where selection occurred
    created_at: datetime

    class Config:
        from_attributes = True


class SelectedTextCreate(BaseModel):
    content: str
    module_id: str
    chapter_id: str
    section_id: str
    hierarchy_path: str


class SelectedTextUpdate(BaseModel):
    content: Optional[str] = None


# For database modeling using SQLAlchemy
from sqlalchemy import Column, DateTime, String, Text, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.sql import func

Base = declarative_base()


class SelectedTextDB(Base):
    __tablename__ = "selected_texts"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    content = Column(Text, nullable=False)  # The actual selected text (minimum 20 characters)
    module_id = Column(String, nullable=False)  # Module where text was selected
    chapter_id = Column(String, nullable=False)  # Chapter where text was selected
    section_id = Column(String, nullable=False)  # Section where text was selected
    hierarchy_path = Column(String, nullable=False)  # Full path in textbook hierarchy where selection occurred
    created_at = Column(DateTime, server_default=func.now())