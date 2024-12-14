import tiktoken
from utils.text_splitter import hierarchical_split
from processors.gpt_processor import extract_sections, extract_requirements
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def process_legal_document(self, text: str) -> Dict:
        """Process legal document through hierarchical splitting and analysis"""
        # Initial splitting into ~5000 token chunks
        initial_chunks = hierarchical_split(text, max_tokens=5000)
        
        # Extract document structure
        sections = []
        for chunk in initial_chunks:
            chunk_sections = extract_sections(chunk)
            sections.extend(chunk_sections)
        
        # Extract requirements and prohibitions
        requirements = []
        prohibitions = []
        for section in sections:
            section_reqs = extract_requirements(section['text'])
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
            chunk_sections = extract_sections(chunk)
            sections.extend(chunk_sections)
        
        return {
            'sections': sections
        }
