from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class Student(BaseModel):
    """
    Primary user of the chatbot system
    """
    id: UUID
    created_at: datetime
    updated_at: datetime
    preferences: Optional[dict] = None
    last_module_accessed: Optional[str] = None

    class Config:
        from_attributes = True


class StudentCreate(BaseModel):
    preferences: Optional[dict] = None
    last_module_accessed: Optional[str] = None


class StudentUpdate(BaseModel):
    preferences: Optional[dict] = None
    last_module_accessed: Optional[str] = None


# For database modeling using SQLAlchemy
from sqlalchemy import Column, DateTime, String, Text, JSON, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.sql import func

Base = declarative_base()


class StudentDB(Base):
    __tablename__ = "students"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    preferences = Column(JSON)
    last_module_accessed = Column(String)