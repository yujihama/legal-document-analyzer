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
                'ja':
                "文書のセクションとその階層構造を抽出してください。'sections'キーを持つJSONで返してください。各セクションは'title'と'text'フィールドを持ちます。回答フォーマット：{'sections':[{'title':'XXXXX','text':'XXXXXXXXXXXX'},...]}",
                'en':
                "Extract the document sections and their hierarchy. Return as JSON with a 'sections' key containing an array of sections. Each section should have 'title' and 'text' fields.format:{'sections':[{'title':'XXXXX','text':'XXXXXXXXXXXX'},...]}"
            },
            'extract_requirements': {
                'ja':
                "テキストから要求事項と禁止事項を抽出してください。'requirements'と'prohibitions'の配列を持つJSONで返してください。各項目は'text'と'source_section'フィールドを持ちます。",
                'en':
                "Extract requirements ('must do') and prohibitions ('must not do') from the text. Return JSON with 'requirements' and 'prohibitions' arrays. Each item should have 'text' and 'source_section' fields."
            },
            'analyze_compliance': {
                'ja':
                "規制が要件を満たしているか分析してください。以下のフィールドを持つJSONで返してください：compliant（真偽値）、score（0から1の数値）、explanation（説明文）。",
                'en':
                "Analyze if the regulation satisfies the requirement. Return JSON with the following fields: compliant (boolean), score (float between 0 and 1), explanation (string)."
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
            messages=[{
                "role": "system",
                "content": prompt
            }, {
                "role": "user",
                "content": text
            }],
            response_format={"type": "json_object"})

        result = json.loads(response.choices[0].message.content)
        debug_info['response'] = result

        # Save debug info to session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {
                'legal': {
                    'debug_info': []
                },
                'internal': {
                    'debug_info': []
                }
            }
        st.session_state.processing_results['legal']['debug_info'].append(
            debug_info)

        if 'sections' not in result:
            # If GPT doesn't return the expected format, create it
            return {'sections': [{'title': 'Section 1', 'text': text}]}
        return result

    def extract_requirements(self,
                             text: str,
                             context: Dict = None,
                             num_extractions: int = 1,
                             threshold: float = 0.6) -> Dict:
        """Extract requirements and prohibitions from text using multiple extractions and majority voting"""
        import streamlit as st

        all_requirements = []
        all_prohibitions = []

        prompt = self.get_prompt('extract_requirements')

        # Perform multiple extractions
        for i in range(num_extractions):
            debug_info = {
                'title':
                f'Extract Requirements (Attempt {i+1}/{num_extractions})',
                'input': f"Text: {text[:500]}...\nPrompt: {prompt}",
                'response': None
            }

            # Prepare context information for prompt
            context_prompt = ""
            if context:
                context_prompt = f"""
文書の種類: {context.get('document_type', '不明')}
主題: {context.get('main_subject', '不明')}
重要な概念: {', '.join(context.get('key_concepts', []))}
現在のセクション: {context.get('local_context', {}).get('current_section', '不明')}
セクションの概要: {context.get('local_context', {}).get('section_summary', '不明')}

上記のコンテキストを考慮して、以下のテキストから要求事項と禁止事項を抽出してください：
"""

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": prompt
                }, {
                    "role": "user",
                    "content": context_prompt + text if context_prompt else text
                }],
                response_format={"type": "json_object"})

            result = json.loads(response.choices[0].message.content)
            debug_info['response'] = result

            # Save debug info to session state
            if 'processing_results' not in st.session_state:
                st.session_state.processing_results = {
                    'legal': {
                        'debug_info': []
                    },
                    'internal': {
                        'debug_info': []
                    }
                }
            st.session_state.processing_results['legal']['debug_info'].append(
                debug_info)

            if 'requirements' in result and 'prohibitions' in result:
                all_requirements.extend(result['requirements'])
                all_prohibitions.extend(result['prohibitions'])

        # Helper function to find similar items
        def find_similar_items(items):
            from collections import defaultdict
            groups = defaultdict(list)

            for item in items:
                text = item['text']
                found_group = False

                # Compare with existing groups
                for key in groups:
                    # Simple similarity check based on common words
                    common_words = set(text.split()) & set(key.split())
                    similarity = len(common_words) / max(
                        len(text.split()), len(key.split()))

                    if similarity > 0.7:  # Threshold for considering items similar
                        groups[key].append(item)
                        found_group = True
                        break

                if not found_group:
                    groups[text].append(item)

            return groups

        # Process requirements and prohibitions
        def process_items(items, threshold_count):
            groups = find_similar_items(items)
            result = []

            for key, group in groups.items():
                if len(group) >= threshold_count:
                    # Use the most common version of the text
                    from collections import Counter
                    texts = Counter(item['text'] for item in group)
                    most_common_text = texts.most_common(1)[0][0]

                    # Use the most detailed source section
                    source_sections = [
                        item.get('source_section', '') for item in group
                    ]
                    source_section = max(source_sections, key=len, default='')

                    result.append({
                        'text': most_common_text,
                        'source_section': source_section
                    })

            return result

        # Calculate threshold count based on number of extractions
        threshold_count = int(num_extractions * threshold)

        # Process both requirements and prohibitions
        final_requirements = process_items(all_requirements, threshold_count)
        final_prohibitions = process_items(all_prohibitions, threshold_count)

        return {
            'requirements': final_requirements,
            'prohibitions': final_prohibitions
        }

    def analyze_compliance(self, requirement: str, regulation: str) -> Dict:
        """Analyze if regulation satisfies requirement"""
        import streamlit as st

        prompt = self.get_prompt('analyze_compliance')
        debug_info = {
            'title': 'Analyze Compliance',
            'input':
            f"Requirement: {requirement}\nRegulation: {regulation}\nPrompt: {prompt}",
            'response': None
        }

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": prompt
            }, {
                "role":
                "user",
                "content":
                f"Requirement: {requirement}\nRegulation: {regulation}"
            }],
            response_format={"type": "json_object"})

        result = json.loads(response.choices[0].message.content)
        debug_info['response'] = result

        # Save debug info to session state
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {
                'legal': {
                    'debug_info': []
                },
                'internal': {
                    'debug_info': []
                }
            }
        st.session_state.processing_results['internal']['debug_info'].append(
            debug_info)

        # Ensure all required fields are present
        if not all(k in result for k in ['compliant', 'score', 'explanation']):
            return {
                'compliant': False,
                'score': 0.0,
                'explanation': 'Failed to analyze compliance'
            }
        return result

    def extract_hierarchical_context(self, text: str) -> Dict:
        """Extract hierarchical context information from the document"""
        prompt = """
        文書から階層的なコンテキスト情報を抽出してください。
        以下の情報を含むJSONで返してください：
        - document_type: 文書の種類（法令、規則、ガイドラインなど）
        - main_subject: 文書の主題
        - key_concepts: 重要な概念や定義のリスト
        - hierarchy: 文書の階層構造（章、節、項など）

        回答フォーマット：
        {
            "document_type": "文書の種類",
            "main_subject": "主題",
            "key_concepts": ["概念1", "概念2", ...],
            "hierarchy": {
                "title": "文書タイトル",
                "sections": [
                    {
                        "title": "章タイトル",
                        "level": 1,
                        "summary": "章の要約",
                        "sections": [...]
                    }
                ]
            }
        }
        """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": prompt
            }, {
                "role": "user",
                "content": text
            }],
            response_format={"type": "json_object"})

        return json.loads(response.choices[0].message.content)

    def generate_report(self, analysis_results: Dict) -> str:
        """Generate compliance report in markdown format"""
        prompts = {
            'ja':
            "以下の分析結果に基づいて、詳細なコンプライアンスレポートをマークダウン形式で生成してください。要求事項の遵守状況、ギャップ、改善提案を含めてください。",
            'en':
            "Generate a detailed compliance report in markdown format based on the analysis results. Include compliance status, gaps, and improvement suggestions."
        }

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": prompts[self.language]
            }, {
                "role": "user",
                "content": json.dumps(analysis_results)
            }])
        return response.choices[0].message.content