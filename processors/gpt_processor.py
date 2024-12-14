import json
from openai import OpenAI
from typing import Dict, List

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

class GPTProcessor:
    def __init__(self, language='ja'):
        self.client = OpenAI()
        self.language = language
        
    def get_prompt(self, key: str) -> str:
        """Get prompt based on language setting"""
        prompts = {
            'extract_sections': {
                'ja': "文書のセクションとその階層構造を抽出してください。'sections'キーを持つJSONで返してください。各セクションは'title'と'text'フィールドを持ちます。",
                'en': "Extract the document sections and their hierarchy. Return as JSON with a 'sections' key containing an array of sections. Each section should have 'title' and 'text' fields."
            },
            'extract_requirements': {
                'ja': "テキストから要求事項（〜しなければならない）と禁止事項（〜してはならない）を抽出してください。'requirements'と'prohibitions'の配列を持つJSONで返してください。各項目は'text'と'source_section'フィールドを持ちます。",
                'en': "Extract requirements ('must do') and prohibitions ('must not do') from the text. Return JSON with 'requirements' and 'prohibitions' arrays. Each item should have 'text' and 'source_section' fields."
            },
            'analyze_compliance': {
                'ja': "規制が要件を満たしているか分析してください。以下のフィールドを持つJSONで返してください：compliant（真偽値）、score（0から1の数値）、explanation（説明文）。",
                'en': "Analyze if the regulation satisfies the requirement. Return JSON with the following fields: compliant (boolean), score (float between 0 and 1), explanation (string)."
            }
        }
        return prompts[key][self.language]
    
    def extract_sections(self, text: str) -> Dict:
        """Extract document sections using GPT-4o"""
        import streamlit as st
        
        prompt = self.get_prompt('extract_sections')
        debug_info = {
            'title': 'Extract Sections',
            'input': f"Text: {text[:500]}...\nPrompt: {prompt}",
            'response': None
        }
        
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        debug_info['response'] = result
        
        # Save debug info to session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {
                'legal': {'debug_info': []},
                'internal': {'debug_info': []}
            }
        st.session_state.processing_results['legal']['debug_info'].append(debug_info)
        
        if 'sections' not in result:
            # If GPT doesn't return the expected format, create it
            return {
                'sections': [
                    {
                        'title': 'Section 1',
                        'text': text
                    }
                ]
            }
        return result
    
    def extract_requirements(self, text: str) -> Dict:
        """Extract requirements and prohibitions from text"""
        import streamlit as st
        
        prompt = self.get_prompt('extract_requirements')
        debug_info = {
            'title': 'Extract Requirements',
            'input': f"Text: {text[:500]}...\nPrompt: {prompt}",
            'response': None
        }
        
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        debug_info['response'] = result
        
        # Save debug info to session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {
                'legal': {'debug_info': []},
                'internal': {'debug_info': []}
            }
        st.session_state.processing_results['legal']['debug_info'].append(debug_info)
        
        # Ensure the response has the required keys
        if 'requirements' not in result or 'prohibitions' not in result:
            return {
                'requirements': [],
                'prohibitions': []
            }
        return result
    
    def analyze_compliance(self, requirement: str, regulation: str) -> Dict:
        """Analyze if regulation satisfies requirement"""
        import streamlit as st
        
        prompt = self.get_prompt('analyze_compliance')
        debug_info = {
            'title': 'Analyze Compliance',
            'input': f"Requirement: {requirement}\nRegulation: {regulation}\nPrompt: {prompt}",
            'response': None
        }
        
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": f"Requirement: {requirement}\nRegulation: {regulation}"
                }
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        debug_info['response'] = result
        
        # Save debug info to session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {
                'legal': {'debug_info': []},
                'internal': {'debug_info': []}
            }
        st.session_state.processing_results['internal']['debug_info'].append(debug_info)
        
        # Ensure all required fields are present
        if not all(k in result for k in ['compliant', 'score', 'explanation']):
            return {
                'compliant': False,
                'score': 0.0,
                'explanation': 'Failed to analyze compliance'
            }
        return result
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate compliance report in markdown format"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a detailed compliance report in markdown format based on the analysis results."
                },
                {
                    "role": "user",
                    "content": json.dumps(analysis_results)
                }
            ]
        )
        return response.choices[0].message.content
