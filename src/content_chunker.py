import math
from typing import List, Dict, Optional
from dataclasses import dataclass
from .content_parser import ParsedContent


@dataclass
class ChunkedContent:
    """
    Data class to represent chunked content ready for embedding
    """
    id: str
    module_id: str
    chapter_id: str
    section_id: str
    hierarchy_path: str
    content_type: str
    content: str
    metadata: Optional[Dict] = None
    original_id: Optional[str] = None  # ID of the original content this chunk came from


class ContentChunker:
    """
    Class to chunk parsed content into appropriate sizes for GEMINI embeddings
    Maintains context boundaries while chunking
    """
    
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        """
        Initialize the content chunker
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks to maintain context
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # We'll use a rough estimation: 1 token ≈ 4 characters for English text
        self.max_chars = max_tokens * 4
        self.overlap_chars = overlap_tokens * 4
    
    def chunk_content(self, parsed_contents: List[ParsedContent]) -> List[ChunkedContent]:
        """
        Chunk the parsed content into appropriate sizes for embeddings
        """
        chunked_contents = []
        
        for parsed_content in parsed_contents:
            # Calculate if the content needs to be chunked
            if len(parsed_content.content) <= self.max_chars:
                # Content fits in a single chunk
                chunked_contents.append(
                    ChunkedContent(
                        id=f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_0",
                        module_id=parsed_content.module_id,
                        chapter_id=parsed_content.chapter_id,
                        section_id=parsed_content.section_id,
                        hierarchy_path=parsed_content.hierarchy_path,
                        content_type=parsed_content.content_type,
                        content=parsed_content.content,
                        metadata=parsed_content.metadata,
                        original_id=parsed_content.section_id
                    )
                )
            else:
                # Content needs to be chunked
                sub_chunks = self._chunk_large_content(parsed_content)
                chunked_contents.extend(sub_chunks)
        
        return chunked_contents
    
    def _chunk_large_content(self, parsed_content: ParsedContent) -> List[ChunkedContent]:
        """
        Chunk large content into smaller pieces
        """
        chunks = []
        content = parsed_content.content
        content_length = len(content)
        
        # Split content by sentences to maintain context boundaries
        sentences = self._split_by_sentences(content)
        
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed the token limit
            if len(current_chunk + sentence) > self.max_chars:
                # If current chunk is not empty, save it
                if current_chunk.strip():
                    chunk_id = f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_{chunk_idx}"
                    chunks.append(
                        ChunkedContent(
                            id=chunk_id,
                            module_id=parsed_content.module_id,
                            chapter_id=parsed_content.chapter_id,
                            section_id=f"{parsed_content.section_id}_chunk{chunk_idx}",
                            hierarchy_path=f"{parsed_content.hierarchy_path}_chunk{chunk_idx}",
                            content_type=parsed_content.content_type,
                            content=current_chunk.strip(),
                            metadata=parsed_content.metadata,
                            original_id=parsed_content.section_id
                        )
                    )
                    chunk_idx += 1
                
                # Start a new chunk with this sentence
                # If the sentence is too long, we'll need to force split it
                if len(sentence) > self.max_chars:
                    # Split the long sentence into smaller parts
                    sentence_chunks = self._force_chunk_sentence(sentence)
                    for i, sentence_chunk in enumerate(sentence_chunks):
                        chunk_id = f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_{chunk_idx}"
                        chunks.append(
                            ChunkedContent(
                                id=chunk_id,
                                module_id=parsed_content.module_id,
                                chapter_id=parsed_content.chapter_id,
                                section_id=f"{parsed_content.section_id}_chunk{chunk_idx}",
                                hierarchy_path=f"{parsed_content.hierarchy_path}_chunk{chunk_idx}",
                                content_type=parsed_content.content_type,
                                content=sentence_chunk,
                                metadata=parsed_content.metadata,
                                original_id=parsed_content.section_id
                            )
                        )
                        chunk_idx += 1
                    
                    # Start a new current chunk after handling the long sentence
                    current_chunk = ""
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                current_chunk += sentence
        
        # Add the final chunk if there's content left
        if current_chunk.strip():
            chunk_id = f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_{chunk_idx}"
            chunks.append(
                ChunkedContent(
                    id=chunk_id,
                    module_id=parsed_content.module_id,
                    chapter_id=parsed_content.chapter_id,
                    section_id=f"{parsed_content.section_id}_chunk{chunk_idx}",
                    hierarchy_path=f"{parsed_content.hierarchy_path}_chunk{chunk_idx}",
                    content_type=parsed_content.content_type,
                    content=current_chunk.strip(),
                    metadata=parsed_content.metadata,
                    original_id=parsed_content.section_id
                )
            )
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving the sentence structure
        """
        # This is a simple sentence splitter; for production, use a proper NLP library
        import re
        
        # Split by sentence endings followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Add back the sentence ending punctuation
        result = []
        for sentence in sentences[:-1]:  # All except the last one
            if sentence.strip():
                # Add the punctuation back
                if sentence[-1] not in '.!?':
                    sentence += '.'
                result.append(sentence + ' ')
        
        # Handle the last sentence
        if sentences and sentences[-1].strip():
            result.append(sentences[-1])
        
        return result
    
    def _force_chunk_sentence(self, sentence: str) -> List[str]:
        """
        Force chunk a sentence that's too long
        """
        if len(sentence) <= self.max_chars:
            return [sentence]
        
        chunks = []
        start = 0
        
        while start < len(sentence):
            end = start + self.max_chars
            
            # Try to find a good breaking point (space or punctuation)
            if end < len(sentence):
                # Look for a space or punctuation to break at
                break_point = -1
                for i in range(end, start, -1):
                    if sentence[i] in ' .,;:!?-':
                        break_point = i + 1
                        break
                
                # If no good break point found, force break
                if break_point == -1:
                    break_point = end
                
                chunks.append(sentence[start:break_point])
                start = break_point
            else:
                chunks.append(sentence[start:])
                break
        
        return chunks
    
    def chunk_by_semantic_boundaries(self, parsed_contents: List[ParsedContent]) -> List[ChunkedContent]:
        """
        Chunk content respecting semantic boundaries like paragraphs, code blocks, etc.
        This is an alternative approach that tries to keep semantically related content together
        """
        chunked_contents = []
        
        for parsed_content in parsed_contents:
            if parsed_content.content_type == "code":
                # For code, try to keep functions/classes together if possible
                sub_chunks = self._chunk_code_content(parsed_content)
                chunked_contents.extend(sub_chunks)
            else:
                # For regular text, use the standard chunking approach
                sub_chunks = self._chunk_large_content(parsed_content)
                chunked_contents.extend(sub_chunks)
        
        return chunked_contents
    
    def _chunk_code_content(self, parsed_content: ParsedContent) -> List[ChunkedContent]:
        """
        Specialized chunking for code content to preserve function/class boundaries
        """
        chunks = []
        content = parsed_content.content
        
        # For code content, try to split by functions/classes
        lines = content.split('\\n')
        
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if adding this line would exceed the character limit
            if current_length + len(line) > self.max_chars and current_chunk:
                # Save current chunk
                chunk_id = f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_code_chunk{chunk_idx}"
                chunks.append(
                    ChunkedContent(
                        id=chunk_id,
                        module_id=parsed_content.module_id,
                        chapter_id=parsed_content.chapter_id,
                        section_id=f"{parsed_content.section_id}_code_chunk{chunk_idx}",
                        hierarchy_path=f"{parsed_content.hierarchy_path}_code_chunk{chunk_idx}",
                        content_type=parsed_content.content_type,
                        content='\\n'.join(current_chunk),
                        metadata=parsed_content.metadata,
                        original_id=parsed_content.section_id
                    )
                )
                chunk_idx += 1
                current_chunk = []
                current_length = 0
            else:
                current_chunk.append(line)
                current_length += len(line) + 1  # +1 for newline character
                i += 1
        
        # Add the final chunk if there's content left
        if current_chunk:
            chunk_id = f"{parsed_content.module_id}_{parsed_content.chapter_id}_{parsed_content.section_id}_code_chunk{chunk_idx}"
            chunks.append(
                ChunkedContent(
                    id=chunk_id,
                    module_id=parsed_content.module_id,
                    chapter_id=parsed_content.chapter_id,
                    section_id=f"{parsed_content.section_id}_code_chunk{chunk_idx}",
                    hierarchy_path=f"{parsed_content.hierarchy_path}_code_chunk{chunk_idx}",
                    content_type=parsed_content.content_type,
                    content='\\n'.join(current_chunk),
                    metadata=parsed_content.metadata,
                    original_id=parsed_content.section_id
                )
            )
        
        return chunks


def chunk_content_for_gemini(parsed_contents: List[ParsedContent], max_tokens: int = 500) -> List[ChunkedContent]:
    """
    Convenience function to chunk content for GEMINI embeddings without creating a full chunker instance
    """
    chunker = ContentChunker(max_tokens=max_tokens)
    return chunker.chunk_content(parsed_contents)


if __name__ == "__main__":
    # Example usage
    from .content_parser import ParsedContent
    
    sample_content = ParsedContent(
        module_id="module1",
        chapter_id="chapter1",
        section_id="section1",
        hierarchy_path="module1/chapter1/section1",
        content_type="text",
        content="This is a sample content that is quite long and will need to be chunked into smaller pieces for processing. " * 30,
        title="Sample Content"
    )
    
    chunker = ContentChunker(max_tokens=100)  # Small tokens for testing
    chunks = chunker.chunk_content([sample_content])
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  ID: {chunk.id}")
        print(f"  Type: {chunk.content_type}")
        print(f"  Length: {len(chunk.content)} characters")
        print(f"  Content preview: {chunk.content[:100]}...")
        print()