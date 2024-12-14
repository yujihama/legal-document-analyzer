import json
from openai import OpenAI
from typing import Dict, List

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

class GPTProcessor:
    def __init__(self):
        self.client = OpenAI()
    
    def extract_sections(self, text: str) -> List[Dict]:
        """Extract document sections using GPT-4o"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Extract the document sections and their hierarchy. Return as JSON."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def extract_requirements(self, text: str) -> Dict:
        """Extract requirements and prohibitions from text"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Extract requirements ('must do') and prohibitions ('must not do') from the text. Return as JSON with two lists."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def analyze_compliance(self, requirement: str, regulation: str) -> Dict:
        """Analyze if regulation satisfies requirement"""
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Analyze if the regulation satisfies the requirement. Return JSON with compliance status and explanation."
                },
                {
                    "role": "user",
                    "content": f"Requirement: {requirement}\nRegulation: {regulation}"
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
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
