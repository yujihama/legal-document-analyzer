import json
import time
from functools import wraps
from openai import RateLimitError, OpenAI
from typing import Dict, List

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
                    print(f"レート制限を超過しました。{wait_time}秒待機してから再試行します。(試行 {retries}/{max_retries})")
                    time.sleep(wait_time)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
MODEL_NAME = "gpt-4o"

class GPTProcessor:
    # Define prompts as class variables
    PROMPTS = {
        'extract_sections': {
            'ja': "文書のセクションとその階層構造を抽出してください。'sections'キーを持つJSONで返してください。各セクションは'title'と'text'フィールドを持ちます。回答フォーマット：{'sections':[{'title':'XXXXX','text':'XXXXXXXXXXXX'},...]}",
            'en': "Extract document sections and hierarchy. Return JSON with 'sections' key."
        },
        'extract_requirements': {
            'ja': "テキストから要求事項と禁止事項を抽出してください。'requirements'と'prohibitions'の配列を持つJSONで返してください。",
            'en': "Extract requirements and prohibitions. Return JSON with 'requirements' and 'prohibitions' arrays."
        },
        'analyze_compliance': {
            'ja': "規制が要件を満たしているか分析してください。JSONで返してください：compliant（真偽値）、score（0-1）、explanation（説明）。",
            'en': "Analyze compliance. Return JSON with compliant (boolean), score (0-1), explanation."
        },
        'report': {
            'summary': {
                'ja': "コンプライアンス状況の概要をJSON形式で生成してください：{\"summary\":{\"overview\":\"状況説明\",\"compliance_rate\":\"遵守率\",\"key_findings\":[\"発見1\",\"発見2\"]}}",
                'en': "Generate compliance overview in JSON format."
            },
            'requirements': {
                'ja': "要件グループの分析結果をJSON形式で返してください：{\"analysis\":{\"requirements\":[{\"overview\":\"概要\",\"status\":\"状況\",\"measures\":\"対応\"}]}}",
                'en': "Analyze requirements group in JSON format."
            },
            'recommendations': {
                'ja': "改善提案をJSON形式で生成してください：{\"recommendations\":{\"actions\":[{\"title\":\"提案\",\"description\":\"説明\",\"priority\":\"優先度\",\"impact\":\"影響\"}]}}",
                'en': "Generate improvement suggestions in JSON format."
            }
        }
    }

    def __init__(self, language='ja'):
        self.client = OpenAI()
        self.language = language

    def get_prompt(self, key: str) -> str:
        """Get prompt based on language setting"""
        if key not in self.PROMPTS:
            raise KeyError(f"Invalid prompt key: {key}")
        
        prompt_data = self.PROMPTS[key]
        if isinstance(prompt_data, dict) and self.language in prompt_data:
            return prompt_data[self.language]
        raise KeyError(f"Language {self.language} not found for key {key}")

    def get_report_prompt(self, section: str) -> str:
        """Get report section prompt based on language setting"""
        if 'report' not in self.PROMPTS or section not in self.PROMPTS['report']:
            raise KeyError(f"Invalid report section: {section}")
        
        if self.language not in self.PROMPTS['report'][section]:
            raise KeyError(f"Language {self.language} not found for section {section}")
        
        return self.PROMPTS['report'][section][self.language]

    @retry_on_rate_limit()
    def extract_sections(self, text: str) -> Dict:
        """Extract document sections using GPT-4"""
        import streamlit as st

        prompt = self.get_prompt('extract_sections')
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
            if 'sections' not in result:
                return {'sections': [{'title': 'Section 1', 'text': text}]}
            return result
        except json.JSONDecodeError:
            return {'sections': [{'title': 'Section 1', 'text': text}]}

    @retry_on_rate_limit()
    def extract_requirements(self, text: str, context: Dict = None) -> Dict:
        """Extract requirements and prohibitions from text"""
        import streamlit as st

        prompt = self.get_prompt('extract_requirements')
        context_prompt = ""
        if context:
            context_prompt = f"""
            文書の種類: {context.get('document_type', '不明')}
            主題: {context.get('main_subject', '不明')}
            重要な概念: {', '.join(context.get('key_concepts', []))}
            """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context_prompt + text if context_prompt else text}
            ],
            response_format={"type": "json_object"}
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {'requirements': [], 'prohibitions': []}

    @retry_on_rate_limit()
    def analyze_compliance(self, requirement: str, regulation: str) -> Dict:
        """Analyze if regulation satisfies requirement"""
        prompt = self.get_prompt('analyze_compliance')
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Requirement: {requirement}\nRegulation: {regulation}"}
            ],
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
            return {
                'compliant': result.get('compliant', False),
                'score': result.get('score', 0.0),
                'explanation': result.get('explanation', 'Analysis failed')
            }
        except json.JSONDecodeError:
            return {
                'compliant': False,
                'score': 0.0,
                'explanation': 'Failed to analyze compliance'
            }

    @retry_on_rate_limit()
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate comprehensive compliance report"""
        try:
            # Calculate statistics
            stats = {
                'total_requirements': len(analysis_results.get('compliance_results', [])),
                'compliant_count': sum(
                    1 for r in analysis_results.get('compliance_results', [])
                    if any(m['analysis']['compliant'] for m in r['matches'])
                ),
            }
            stats['compliance_rate'] = (stats['compliant_count'] / stats['total_requirements'] * 100) if stats['total_requirements'] > 0 else 0

            # Generate overview
            overview_response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.get_report_prompt('summary')},
                    {"role": "user", "content": json.dumps(stats, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"}
            )

            overview_data = json.loads(overview_response.choices[0].message.content)
            summary = overview_data.get('summary', {})

            # Format report
            report = [
                "# コンプライアンス分析レポート\n",
                "## 概要",
                summary.get('overview', '概要情報なし'),
                "\n### 遵守率",
                summary.get('compliance_rate', '遵守率情報なし'),
                "\n### 主要な発見事項"
            ]

            for finding in summary.get('key_findings', ['分析結果がありません']):
                report.append(f"- {finding}")

            return '\n'.join(report)

        except Exception as e:
            return f"# コンプライアンス分析レポート\n\n## エラー\nレポートの生成中にエラーが発生しました：{str(e)}"

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