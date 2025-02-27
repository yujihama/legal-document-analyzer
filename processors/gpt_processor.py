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
    def summarize_cluster_requirements(self, requirements: List[Dict], prohibitions: List[Dict]) -> Dict:
        """Generate a comprehensive summary of requirements and prohibitions in a cluster"""
        prompts = {
            'ja': """
            以下の要件と禁止事項から、包括的な要約を生成してください。
            要件と禁止事項の意図を保持しながら、重複を排除し、論理的にまとめてください。

            回答は以下のJSON形式で返してください：
            {
                "comprehensive_summary": "すべての要件と禁止事項を網羅した要約文",
                "key_points": ["重要なポイント1", "重要なポイント2", ...],
                "requirements_summary": "要件のまとめ",
                "prohibitions_summary": "禁止事項のまとめ"
            }
            """,
            'en': """
            Generate a comprehensive summary from the following requirements and prohibitions.
            Maintain the intent while eliminating duplicates and organizing logically.

            Please return in the following JSON format:
            {
                "comprehensive_summary": "summary covering all requirements and prohibitions",
                "key_points": ["key point 1", "key point 2", ...],
                "requirements_summary": "summary of requirements",
                "prohibitions_summary": "summary of prohibitions"
            }
            """
        }

        # Prepare input text
        input_text = "要件:\n" + "\n".join([f"- {r['text']}" for r in requirements])
        input_text += "\n\n禁止事項:\n" + "\n".join([f"- {p['text']}" for p in prohibitions])

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": prompts[self.language]
            }, {
                "role": "user",
                "content": input_text
            }],
            response_format={"type": "json_object"})
        
        return json.loads(response.choices[0].message.content)

    @retry_on_rate_limit()
    def analyze_cluster_compliance(self, cluster_summary: str, regulations: List[str], num_trials: int = 3) -> Dict:
        """Analyze if regulations satisfy the cluster's requirements summary and aggregate results from multiple trials"""
        prompts = {
            'ja': """
            クラスタの要約と社内規定を比較し、包括的なコンプライアンス評価を行ってください。
            
            回答は以下のJSON形式で返してください：
            {
                "overall_compliance": true/false,
                "compliance_score": 0.0-1.0,
                "analysis": "詳細な分析",
                "key_findings": ["発見1", "発見2", ...],
                "improvement_suggestions": ["提案1", "提案2", ...]
            }
            """,
            'en': """
            Compare the cluster summary with internal regulations and perform a comprehensive compliance evaluation.
            
            Please return in the following JSON format:
            {
                "overall_compliance": true/false,
                "compliance_score": 0.0-1.0,
                "analysis": "detailed analysis",
                "key_findings": ["finding 1", "finding 2", ...],
                "improvement_suggestions": ["suggestion 1", "suggestion 2", ...]
            }
            """
        }

        # Prepare input text
        input_text = f"クラスタ要約:\n{cluster_summary}\n\n社内規定:\n" + "\n\n".join(regulations)

        try:
            # 複数回の判定を実行
            trial_results = []
            for _ in range(num_trials):
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "system",
                        "content": prompts[self.language]
                    }, {
                        "role": "user",
                        "content": input_text
                    }],
                    response_format={"type": "json_object"})
                
                result = json.loads(response.choices[0].message.content)
                trial_results.append(result)

            # 結果の統計的集約
            compliant_count = sum(1 for r in trial_results if r.get("overall_compliance", False))
            avg_score = sum(float(r.get("compliance_score", 0.0)) for r in trial_results) / num_trials
            
            # 多数決による判定
            final_compliance = compliant_count > (num_trials / 2)
            
            # 分析コメントの集約
            all_findings = [finding for r in trial_results for finding in r.get("key_findings", [])]
            all_suggestions = [sugg for r in trial_results for sugg in r.get("improvement_suggestions", [])]
            
            # 重複を除去
            unique_findings = list(dict.fromkeys(all_findings))
            unique_suggestions = list(dict.fromkeys(all_suggestions))
            
            return {
                "overall_compliance": final_compliance,
                "compliance_score": avg_score,
                "analysis": f"複数回の判定結果（{compliant_count}/{num_trials}が遵守と判定）に基づく分析です。",
                "key_findings": unique_findings[:5],  # 上位5件まで
                "improvement_suggestions": unique_suggestions[:5],  # 上位5件まで
                "trial_count": num_trials,
                "compliant_trials": compliant_count
            }
        except Exception as e:
            print(f"Error in analyze_cluster_compliance: {e}")
            return {
                "overall_compliance": False,
                "compliance_score": 0.0,
                "analysis": "分析中にエラーが発生しました",
                "key_findings": ["エラーが発生しました"],
                "improvement_suggestions": ["システム管理者に連絡してください"]
            }