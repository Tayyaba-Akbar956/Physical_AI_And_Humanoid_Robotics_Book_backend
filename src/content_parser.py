import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import html


@dataclass
class ParsedContent:
    """
    Data class to represent parsed content
    """
    module_id: str
    chapter_id: str
    section_id: str
    hierarchy_path: str
    content_type: str  # text, code, diagram_description, etc.
    content: str
    title: str = ""
    metadata: Optional[Dict] = None


class ContentParser:
    """
    Class to parse HTML content from the textbook website
    Handles HTML, code blocks, images as descriptions, and preserves hierarchy
    """
    
    def __init__(self):
        self.metadata = {
            "word_count": 0,
            "reading_level": "",
            "content_length": 0,
            "has_code_blocks": False,
            "has_images": False,
            "has_diagrams": False
        }
    
    def parse_html_content(self, html_content: str, hierarchy_path: str) -> List[ParsedContent]:
        """
        Parse HTML content and return structured content chunks
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract path components
        path_parts = hierarchy_path.strip('/').split('/')
        if len(path_parts) >= 3:
            module_id, chapter_id, section_id = path_parts[0], path_parts[1], path_parts[2]
        else:
            module_id = chapter_id = section_id = "unknown"
        
        # Remove script tags and other non-content elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Parse different content types
        parsed_contents = []
        
        # Parse main sections
        sections = soup.find_all(['section', 'article', 'div'], recursive=False)
        
        for i, section in enumerate(sections):
            section_title = ""
            # Look for heading in this section
            heading = section.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if heading:
                section_title = heading.get_text().strip()
            
            # Extract text content from the section
            text_content = self._extract_text_content(section)
            
            if text_content.strip():
                parsed_contents.append(
                    ParsedContent(
                        module_id=module_id,
                        chapter_id=chapter_id,
                        section_id=f"{section_id}-s{i+1}" if section_id != "unknown" else f"s{i+1}",
                        hierarchy_path=f"{hierarchy_path}/s{i+1}",
                        content_type="text",
                        content=text_content,
                        title=section_title
                    )
                )
        
        # If no sections, parse the entire content as one chunk
        if not parsed_contents:
            full_content = self._extract_text_content(soup)
            if full_content.strip():
                parsed_contents.append(
                    ParsedContent(
                        module_id=module_id,
                        chapter_id=chapter_id,
                        section_id=section_id,
                        hierarchy_path=hierarchy_path,
                        content_type="text",
                        content=full_content
                    )
                )
        
        # Parse code blocks separately
        code_contents = self._parse_code_blocks(soup, module_id, chapter_id, section_id, hierarchy_path)
        parsed_contents.extend(code_contents)
        
        # Parse image descriptions
        image_contents = self._parse_image_descriptions(soup, module_id, chapter_id, section_id, hierarchy_path)
        parsed_contents.extend(image_contents)
        
        return parsed_contents
    
    def _extract_text_content(self, element) -> str:
        """
        Extract text content from an HTML element, removing code blocks temporarily
        """
        # Temporarily remove code blocks to avoid parsing their content as text
        code_blocks = element.find_all(['code', 'pre'])
        preserved_codes = []
        
        for i, code in enumerate(code_blocks):
            preserved_codes.append(str(code))
            code.replace_with(f"{{CODE_BLOCK_{i}}}")
        
        # Get text content
        text = element.get_text(separator=' ')
        
        # Restore code blocks
        for i, code in enumerate(preserved_codes):
            text = text.replace(f"{{CODE_BLOCK_{i}}}", code)
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _parse_code_blocks(self, soup, module_id: str, chapter_id: str, section_id: str, hierarchy_path: str) -> List[ParsedContent]:
        """
        Parse code blocks from the HTML content
        """
        code_contents = []
        code_blocks = soup.find_all(['pre', 'code'])
        
        for i, code_block in enumerate(code_blocks):
            code_text = code_block.get_text()
            if code_text.strip():
                code_contents.append(
                    ParsedContent(
                        module_id=module_id,
                        chapter_id=chapter_id,
                        section_id=f"{section_id}-code{i+1}",
                        hierarchy_path=f"{hierarchy_path}/code{i+1}",
                        content_type="code",
                        content=code_text,
                        title=f"Code Block {i+1}"
                    )
                )
        
        return code_contents
    
    def _parse_image_descriptions(self, soup, module_id: str, chapter_id: str, section_id: str, hierarchy_path: str) -> List[ParsedContent]:
        """
        Parse image descriptions from the HTML content
        """
        image_contents = []
        images = soup.find_all('img')
        
        for i, img in enumerate(images):
            alt_text = img.get('alt', '')
            title_text = img.get('title', '')
            src = img.get('src', '')
            
            # Create a description of the image
            description = f"Image: {alt_text or title_text or f'Image at {src}'}"
            
            image_contents.append(
                ParsedContent(
                    module_id=module_id,
                    chapter_id=chapter_id,
                    section_id=f"{section_id}-img{i+1}",
                    hierarchy_path=f"{hierarchy_path}/img{i+1}",
                    content_type="diagram_description",
                    content=description,
                    title=f"Image Description {i+1}"
                )
            )
        
        return image_contents
    
    def parse_textbook_content(self, content: str, hierarchy_path: str) -> List[ParsedContent]:
        """
        Main method to parse textbook content with proper structure preservation
        """
        # This method handles the main content parsing logic
        # It will split content into appropriate chunks while preserving context
        parsed_chunks = self.parse_html_content(content, hierarchy_path)
        
        # Update metadata
        self._update_metadata(parsed_chunks)
        
        return parsed_chunks
    
    def _update_metadata(self, parsed_contents: List[ParsedContent]):
        """
        Update metadata based on parsed content
        """
        total_words = 0
        has_code = False
        has_images = False
        has_diagrams = False
        
        for content in parsed_contents:
            # Count words
            total_words += len(content.content.split())
            
            # Check content types
            if content.content_type == "code":
                has_code = True
            elif content.content_type in ["diagram_description", "image"]:
                has_diagrams = True
        
        self.metadata.update({
            "word_count": total_words,
            "content_length": len(parsed_contents),
            "has_code_blocks": has_code,
            "has_images": has_images,
            "has_diagrams": has_diagrams
        })
    
    def get_reading_level(self, text: str) -> str:
        """
        Calculate reading level (simplified implementation)
        This is a basic implementation; a real solution would use more sophisticated algorithms
        """
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return "N/A"
        
        avg_word_length = sum(len(word) for word in words) / word_count
        sentence_count = len(re.split(r'[.!?]+', text)) or 1
        
        # Very basic reading level calculation
        if avg_word_length < 4 and sentence_count < 15:
            return "Beginner"
        elif avg_word_length < 5 and sentence_count < 20:
            return "Intermediate"
        else:
            return "Advanced"


def parse_content_chunk(content: str, hierarchy_path: str) -> List[ParsedContent]:
    """
    Convenience function to parse content without creating a full parser instance
    """
    parser = ContentParser()
    return parser.parse_textbook_content(content, hierarchy_path)


if __name__ == "__main__":
    # Example usage
    sample_html = """
    <html>
    <body>
        <h1>Introduction to Robotics</h1>
        <section>
            <h2>What is Robotics?</h2>
            <p>Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, computer science, and others.</p>
            <pre><code>def hello_robot():
    print("Hello, Robot!")
            </code></pre>
            <img src="robot.jpg" alt="A simple robot illustration">
        </section>
        <section>
            <h2>History of Robotics</h2>
            <p>The term 'robot' was first used in a 1920 play called R.U.R. (Rossum's Universal Robots).</p>
        </section>
    </body>
    </html>
    """
    
    parser = ContentParser()
    chunks = parser.parse_textbook_content(sample_html, "module1/introduction/chapter1")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Type: {chunk.content_type}")
        print(f"  Module: {chunk.module_id}")
        print(f"  Chapter: {chunk.chapter_id}")
        print(f"  Section: {chunk.section_id}")
        print(f"  Hierarchy: {chunk.hierarchy_path}")
        print(f"  Content: {chunk.content[:100]}...")
        print()