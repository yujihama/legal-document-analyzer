import re
from typing import List
import tiktoken

def hierarchical_split(text: str, max_tokens: int = 5000) -> List[str]:
    """Split text hierarchically respecting semantic boundaries"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    
    def split_on_pattern(text: str, patterns: List[str]) -> List[str]:
        for pattern in patterns:
            if re.search(pattern, text):
                return [chunk.strip() for chunk in re.split(pattern, text) if chunk.strip()]
        return [text]
    
    # Splitting patterns from largest to smallest semantic units
    patterns = [
        r'\n(?=第[一二三四五六七八九十百]+章)',  # Chapters
        r'\n(?=第[一二三四五六七八九十百]+条)',  # Articles
        r'[。．.]\s+',  # Sentences
        r'[、，,]\s+'   # Phrases
    ]
    
    def recursive_split(text: str, pattern_index: int = 0) -> List[str]:
        if count_tokens(text) <= max_tokens:
            return [text]
        
        if pattern_index >= len(patterns):
            # Fallback to character-based splitting if no patterns match
            words = text.split()
            chunks = []
            current_chunk = []
            
            for word in words:
                if count_tokens(" ".join(current_chunk + [word])) > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                else:
                    current_chunk.append(word)
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            return chunks
        
        chunks = split_on_pattern(text, [patterns[pattern_index]])
        
        if len(chunks) == 1:
            return recursive_split(text, pattern_index + 1)
        
        result = []
        for chunk in chunks:
            if count_tokens(chunk) > max_tokens:
                result.extend(recursive_split(chunk, pattern_index + 1))
            else:
                result.append(chunk)
        
        return result
    
    return recursive_split(text)
