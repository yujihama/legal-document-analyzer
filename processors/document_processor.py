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
        """Process legal document through hierarchical splitting and analysis with context"""
        # First, extract the document's hierarchical context
        doc_context = self.gpt_processor.extract_hierarchical_context(text)
        
        # Initial splitting into ~5000 token chunks
        initial_chunks = hierarchical_split(text, max_tokens=5000)
        
        # Extract document structure with context
        sections = []
        current_context = {
            'document_type': doc_context['document_type'],
            'main_subject': doc_context['main_subject'],
            'key_concepts': doc_context['key_concepts']
        }
        
        # Process each chunk with its context
        for i, chunk in enumerate(initial_chunks):
            # Get the relevant hierarchical context for this chunk
            chunk_context = self._get_chunk_context(doc_context, i, len(initial_chunks))
            
            # Combine both document and chunk-specific context
            combined_context = {**current_context, 'local_context': chunk_context}
            
            # Extract sections with context
            chunk_sections = self.gpt_processor.extract_sections(chunk)
            for section in chunk_sections['sections']:
                section['context'] = combined_context
            sections.extend(chunk_sections['sections'])
        
        # Extract requirements and prohibitions with context
        requirements = []
        prohibitions = []
        for section in sections:
            section_context = section.get('context', {})
            section_reqs = self.gpt_processor.extract_requirements(
                section['text'],
                context=section_context
            )
            
            # Add context information to each requirement and prohibition
            for req in section_reqs['requirements']:
                req['context'] = section_context
            for prob in section_reqs['prohibitions']:
                prob['context'] = section_context
            
            requirements.extend(section_reqs['requirements'])
            prohibitions.extend(section_reqs['prohibitions'])
        
        return {
            'sections': sections,
            'requirements': requirements,
            'prohibitions': prohibitions,
            'document_context': doc_context
        }
        
    def _get_chunk_context(self, doc_context: Dict, chunk_index: int, total_chunks: int) -> Dict:
        """Extract relevant context for a specific chunk based on its position"""
        hierarchy = doc_context.get('hierarchy', {})
        
        # Calculate the approximate position in the document
        position = chunk_index / total_chunks
        
        # Find the relevant section in the hierarchy based on position
        current_section = None
        if 'sections' in hierarchy:
            section_index = int(position * len(hierarchy['sections']))
            if section_index < len(hierarchy['sections']):
                current_section = hierarchy['sections'][section_index]
        
        return {
            'position': f"{chunk_index + 1}/{total_chunks}",
            'current_section': current_section['title'] if current_section else None,
            'section_summary': current_section.get('summary', '') if current_section else None
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
