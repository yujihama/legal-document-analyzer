import json
import time
from functools import wraps
from openai import RateLimitError

def retry_on_rate_limit(max_retries=3, wait_time=60):
    """Decorator to retry function on RateLimitError with waiting period"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {retries}/{max_retries}")
                    time.sleep(wait_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator

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

    @retry_on_rate_limit()
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

    @retry_on_rate_limit()
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
            st.text(context_prompt + text if context_prompt else text)

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

    @retry_on_rate_limit()
    @retry_on_rate_limit()
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

    @retry_on_rate_limit()
    @retry_on_rate_limit()
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

    @retry_on_rate_limit()
    @retry_on_rate_limit()
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate compliance report in markdown format by processing chunks of data"""
        
        def get_section_prompt(section_type: str) -> str:
            prompts = {
                'summary': {
                    'ja': "以下の統計情報に基づいて、コンプライアンス状況の概要を生成してください：\n{stats}",
                    'en': "Generate a compliance overview based on the following statistics:\n{stats}"
                },
                'requirements': {
                    'ja': "以下の要件グループについて分析してください（{start}から{end}まで）：\n{reqs}",
                    'en': "Analyze the following group of requirements ({start} to {end}):\n{reqs}"
                },
                'recommendations': {
                    'ja': "分析結果に基づいて、主な改善提案を生成してください：\n{gaps}",
                    'en': "Generate key improvement suggestions based on the analysis:\n{gaps}"
                }
            }
            return prompts[section_type][self.language]
        
        # Generate overview section
        stats = {
            'total_requirements': len(analysis_results.get('compliance_results', [])),
            'compliant_count': sum(1 for r in analysis_results.get('compliance_results', [])
                                 if any(m['analysis']['compliant'] for m in r['matches'])),
            'document_info': {
                'legal': analysis_results.get('documents', {}).get('legal', ''),
                'internal': analysis_results.get('documents', {}).get('internal', '')
            }
        }
        
        overview_response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": get_section_prompt('summary')
            }, {
                "role": "user",
                "content": json.dumps(stats)
            }]
        )
        
        overview_section = overview_response.choices[0].message.content
        
        # Process requirements in chunks
        requirements_sections = []
        chunk_size = 5  # Process 5 requirements at a time
        compliance_results = analysis_results.get('compliance_results', [])
        
        for i in range(0, len(compliance_results), chunk_size):
            chunk = compliance_results[i:i + chunk_size]
            
            req_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "system",
                    "content": get_section_prompt('requirements').format(
                        start=i+1,
                        end=min(i+chunk_size, len(compliance_results)),
                        reqs=json.dumps(chunk)
                    )
                }, {
                    "role": "user",
                    "content": json.dumps(chunk)
                }]
            )
            
            requirements_sections.append(req_response.choices[0].message.content)
        
        # Generate recommendations based on non-compliant items
        gaps = [r for r in compliance_results 
               if not any(m['analysis']['compliant'] for m in r['matches'])]
        
        recommendations_response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": get_section_prompt('recommendations')
            }, {
                "role": "user",
                "content": json.dumps(gaps[:5])  # Process top 5 gaps
            }]
        )
        
        # Combine all sections using string concatenation instead of f-string
        report_parts = [
            "# コンプライアンス分析レポート\n\n",
            "## 概要\n",
            overview_section + "\n\n",
            "## 詳細分析\n",
            "\n\n".join(requirements_sections) + "\n\n",
            "## 改善提案\n",
            recommendations_response.choices[0].message.content
        ]
        report = "".join(report_parts)
        
        return report

    @retry_on_rate_limit()
    @retry_on_rate_limit()
    def summarize_cluster(self, texts: str) -> Dict:
        """Summarize a cluster of texts and generate a representative text"""
        prompts = {
            'ja': """
            以下の関連文書群から、以下の2つを生成してください：
            1. この文書群を代表する代表的なテキスト（最も特徴的な1つを選ぶか、複数を組み合わせて生成）
            2. 文書群全体の要約（トピックや主要なポイントを含む）

            回答は以下のJSON形式で返してください：
            {
                "representative_text": "代表的なテキスト",
                "summary": "全体の要約"
            }
            """,
            'en': """
            From the following related documents, please generate:
            1. A representative text for this document cluster (select the most characteristic one or combine multiple)
            2. A summary of the entire document cluster (including topics and key points)

            Please return in the following JSON format:
            {
                "representative_text": "representative text",
                "summary": "overall summary"
            }
            """
        }

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": prompts[self.language]
            }, {
                "role": "user",
                "content": texts
            }],
            response_format={"type": "json_object"})
        
        return json.loads(response.choices[0].message.content)