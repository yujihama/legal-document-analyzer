import json
from openai import OpenAI
from typing import Dict, List

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

class GPTProcessor:
    def __init__(self):
        self.client = OpenAI()
    
    def extract_sections(self, text: str) -> Dict:
        """Extract document sections using GPT-4o"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the document sections and their hierarchy. Return as JSON with a 'sections' key containing an array of sections. Each section should have 'title' and 'text' fields."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
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
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Extract requirements ('must do') and prohibitions ('must not do') from the text. Return JSON with 'requirements' and 'prohibitions' arrays. Each item should have 'text' and 'source_section' fields."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        # Ensure the response has the required keys
        if 'requirements' not in result or 'prohibitions' not in result:
            return {
                'requirements': [],
                'prohibitions': []
            }
        return result
    
    def analyze_compliance(self, requirement: str, regulation: str) -> Dict:
        """Analyze if regulation satisfies requirement"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Analyze if the regulation satisfies the requirement. Return JSON with the following fields: compliant (boolean), score (float between 0 and 1), explanation (string)."
                },
                {
                    "role": "user",
                    "content": f"Requirement: {requirement}\nRegulation: {regulation}"
                }
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
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
