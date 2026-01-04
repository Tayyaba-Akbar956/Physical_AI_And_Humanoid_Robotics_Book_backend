from datetime import datetime
from typing import Optional
from pydantic import BaseModel
from uuid import UUID, uuid4


class VectorIndexEntry(BaseModel):
    """
    Entry in the vector database for semantic search
    """
    id: UUID
    text_content_id: UUID  # References the Textbook Content entity
    text_chunk: str  # The text that was embedded
    embedding_vector: list  # The actual embedding vector (dimensions based on GEMINI embedding model)
    metadata: Optional[dict] = None  # Metadata for filtering (module, chapter, etc.)

    class Config:
        from_attributes = True


class VectorIndexEntryCreate(BaseModel):
    text_content_id: UUID
    text_chunk: str
    embedding_vector: list
    metadata: Optional[dict] = None


class VectorIndexEntryUpdate(BaseModel):
    text_chunk: Optional[str] = None
    embedding_vector: Optional[list] = None
    metadata: Optional[dict] = None


# This model is primarily for documentation as the vector entries will be stored in Qdrant
# In a real implementation, we would have a service that manages these entries in Qdrant
class VectorIndexEntryDB:
    """
    This is a conceptual model since vector entries are stored in Qdrant, not in Postgres.
    In the actual implementation, we would use the qdrant-client to manage vector entries.
    """
    def __init__(self, id: UUID, text_content_id: UUID, text_chunk: str, 
                 embedding_vector: list, metadata: Optional[dict] = None):
        self.id = id
        self.text_content_id = text_content_id
        self.text_chunk = text_chunk
        self.embedding_vector = embedding_vector
        self.metadata = metadata