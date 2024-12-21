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
    
    def process_legal_document(self, text: str, document_id: str = None) -> Dict:
        """Process legal document and extract requirements directly"""
        from utils.persistence import load_processing_results, save_processing_results
        
        # Generate document_id if not provided
        if document_id is None:
            document_id = str(hash(text))[:8]
            
        # Try to load existing results
        cached_results = load_processing_results(f"legal_doc_{document_id}.json")
        if cached_results:
            print(f"Loading cached results for document {document_id}")
            return cached_results
            
        # Extract document context first
        doc_context = self.gpt_processor.extract_hierarchical_context(text)
        
        # Split document into manageable chunks
        chunks = hierarchical_split(text, max_tokens=5000)
        
        # Process each chunk directly for requirements
        requirements = []
        prohibitions = []
        
        for chunk in chunks:
            # Extract requirements and prohibitions directly from chunk
            chunk_results = self.gpt_processor.extract_requirements(
                chunk,
                context={'document_type': doc_context['document_type']}
            )
            
            # Add extracted items with minimal context
            requirements.extend([
                {'text': req, 'context': {'chunk': chunk}}
                for req in chunk_results['requirements']
            ])
            prohibitions.extend([
                {'text': prob, 'context': {'chunk': chunk}}
                for prob in chunk_results['prohibitions']
            ])
        
        results = {
            'document_id': document_id,
            'requirements': requirements,
            'prohibitions': prohibitions,
            'document_context': doc_context,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save results
        save_processing_results(results, f"legal_doc_{document_id}.json")
        
        return results
        
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
        """Process internal regulations document directly"""
        # Split document into manageable chunks
        chunks = hierarchical_split(text, max_tokens=5000)
        
        # Store chunks directly without section extraction
        return {
            'chunks': chunks
        }
