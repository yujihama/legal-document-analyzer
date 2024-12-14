import tiktoken
from utils.text_splitter import hierarchical_split
from processors.gpt_processor import GPTProcessor
from typing import List, Dict

class DocumentProcessor:
    def __init__(self, language='ja'):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.gpt_processor = GPTProcessor(language=language)
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def process_legal_document(self, text: str) -> Dict:
        """Process legal document through hierarchical splitting and analysis"""
        # Initial splitting into ~5000 token chunks
        initial_chunks = hierarchical_split(text, max_tokens=5000)
        
        # Extract document structure
        sections = []
        for chunk in initial_chunks:
            chunk_sections = self.gpt_processor.extract_sections(chunk)
            sections.extend(chunk_sections['sections'])
        
        # Extract requirements and prohibitions
        requirements = []
        prohibitions = []
        for section in sections:
            section_reqs = self.gpt_processor.extract_requirements(section['text'])
            requirements.extend(section_reqs['requirements'])
            prohibitions.extend(section_reqs['prohibitions'])
        
        return {
            'sections': sections,
            'requirements': requirements,
            'prohibitions': prohibitions
        }
    
    def process_internal_document(self, text: str) -> Dict:
        """Process internal regulations document"""
        # Similar process but focused on matching structure
        initial_chunks = hierarchical_split(text, max_tokens=5000)
        sections = []
        
        for chunk in initial_chunks:
            chunk_sections = self.gpt_processor.extract_sections(chunk)
            sections.extend(chunk_sections['sections'])
        
        return {
            'sections': sections
        }
