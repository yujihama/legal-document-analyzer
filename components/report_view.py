import streamlit as st
from processors.gpt_processor import GPTProcessor
import json
import os
from models.compliance_model import ComplianceReport
from datetime import datetime
import plotly.graph_objects as go
from utils.pdf_generator import PDFReportGenerator


def render_report_section():
    # st.header("コンプライアンスレポート")

    if not st.session_state.get('compliance_results'):
        st.info("先に分析を完了してください")
        return

    # Check if we have a cached report for the current files
    cache_key = None
    if 'documents' in st.session_state:
        import hashlib
        # Generate a unique key based on the content of both documents
        legal_hash = hashlib.md5(st.session_state.documents['legal'].encode()).hexdigest()
        internal_hash = hashlib.md5(st.session_state.documents['internal'].encode()).hexdigest()
        cache_key = f"report_cache_{legal_hash}_{internal_hash}.json"
    
    report_generated = False
    if cache_key and os.path.exists(cache_key):
        try:
            with open(cache_key, 'r', encoding='utf-8') as f:
                import json
                cached_data = json.load(f)
                st.session_state.generated_report = ComplianceReport.from_dict(cached_data)
                report_generated = True
                st.info("既存の分析結果を読み込みました。")
        except Exception as e:
            st.warning(f"キャッシュの読み込みに失敗しました: {str(e)}")
    
    if not report_generated:
        with st.spinner("新しい分析レポートを生成中..."):
            st.session_state.generated_report = generate_compliance_report()
            # Save the report to cache if we have a cache key
            if cache_key:
                try:
                    with open(cache_key, 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.generated_report.to_dict(), f, ensure_ascii=False, indent=2)
                except Exception as e:
                    st.warning(f"キャッシュの保存に失敗しました: {str(e)}")

    display_report(st.session_state.generated_report)


def generate_compliance_report() -> ComplianceReport:
    """Generate comprehensive compliance report based on cluster analysis"""
    gpt_processor = GPTProcessor()

    if not st.session_state.get('compliance_results'):
        raise ValueError("分析結果が見つかりません")

    clusters = st.session_state.compliance_results

    # Count total requirements and compliant clusters
    total_clusters = len(clusters)
    compliant_clusters = sum(
        1 for cluster in clusters
        if cluster.get('analysis', {}).get('overall_compliance', False))

    # Get unique requirements and prohibitions from original analysis results
    if 'analysis_results' in st.session_state:
        results = st.session_state.analysis_results
        total_requirements = len(results['legal']['requirements'])
        total_prohibitions = len(results['legal']['prohibitions'])
    else:
        total_requirements = 0
        total_prohibitions = 0

    # Generate summary content
    summary_parts = [
        "# クラスタベース分析レポート\n", f"## 概要\n",
        f"以下の分析結果は、{total_clusters}個のクラスタに基づいています：\n",
        f"- 分析対象の要件総数: {total_requirements}件",
        f"- 分析対象の禁止事項総数: {total_prohibitions}件",
        f"- 遵守クラスタ数: {compliant_clusters}個",
        f"- 全体遵守率: {(compliant_clusters / total_clusters * 100):.1f}% ({compliant_clusters}/{total_clusters})",
        "\nこの分析結果は、各要件と禁止事項を意味的に関連するグループ（クラスタ）に分類し、",
        "各クラスタごとの遵守状況を総合的に評価したものです。\n", "\n## クラスタ別分析結果\n"
    ]

    # Add cluster-specific summaries
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', 'Unknown')
        analysis = cluster.get('analysis', {})
        summary = cluster.get('summary', {})
        requirements = cluster.get('requirements', [])
        prohibitions = cluster.get('prohibitions', [])

        # Calculate cluster-specific metrics
        req_count = len(requirements)
        prob_count = len(prohibitions)
        compliance_score = analysis.get('compliance_score', 0.0)

        summary_parts.extend([
            f"\n### クラスタ {cluster_id}", f"**基本情報:**",
            f"- 遵守状況: {'遵守' if analysis.get('overall_compliance', False) else '未遵守'}",
            f"- スコア: {compliance_score:.2f}", f"- 含まれる要件数: {req_count}",
            f"- 含まれる禁止事項数: {prob_count}",
            f"\n**要約:**\n{summary.get('comprehensive_summary', '要約なし')}\n",
            "\n**対象となる主な要件:**"
        ])

        # Add key requirements
        for req in requirements[:3]:  # Show top 3 requirements
            summary_parts.append(f"- {req['text']}")

        if len(requirements) > 3:
            summary_parts.append(f"- その他 {len(requirements) - 3} 件の要件\n")

        summary_parts.append("\n**主要な発見事項:**")
        # Add key findings with more context
        for finding in analysis.get('key_findings', ['発見事項なし']):
            if "発見事項なし" not in finding:
                summary_parts.append(f"- {finding}")

        # Add analysis context
        if analysis.get('analysis'):
            summary_parts.extend(["\n**分析コンテキスト:**", analysis.get('analysis')])

    report_content = '\n'.join(summary_parts)

    # Create report object with updated structure
    report = ComplianceReport(
        timestamp=datetime.now(),
        legal_document_name="法令文書",
        internal_document_name="社内規定",
        total_requirements=total_requirements,
        compliant_count=compliant_clusters,
        non_compliant_count=total_clusters - compliant_clusters,
        matches=[],  # 新しい構造では使用しない
        gaps=[],  # 新しい構造では使用しない
        summary=report_content,
        recommendations=[
            cluster.get('analysis', {}).get('improvement_suggestions', [])
            for cluster in clusters
        ])

    return report


def display_report(report: ComplianceReport):
    """Display the cluster-based compliance report in the UI"""
    # ヘッダー情報
    #st.markdown("# コンプライアンス分析レポート")
    st.header("評価結果レポート")
    st.markdown(f"**生成日時:** {report.timestamp.strftime('%Y年%m月%d日 %H:%M')}")

    # エグゼクティブサマリー
    st.markdown("## サマリー")
    col1, col2, col3 = st.columns(3)

    # 要件・禁止事項の総数を計算
    total_items = report.total_requirements + report.non_compliant_count

    with col1:
        st.metric("要件・禁止事項の総数", total_items)
    with col2:
        st.metric("遵守クラスタ数", report.compliant_count)
    with col3:
        compliance_rate = (report.compliant_count / (report.compliant_count + report.non_compliant_count) * 100) \
            if (report.compliant_count + report.non_compliant_count) > 0 else 0
        st.metric("遵守率", f"{compliance_rate:.1f}%")

    # 全体評価
    st.markdown("### 全体評価")
    # サマリーテキストをより堅牢に処理
    summary_text = report.summary
    try:
        if '# クラスタベース分析レポート' in summary_text:
            summary_text = summary_text.split('# クラスタベース分析レポート')[1]
        if '## クラスタ別分析結果' in summary_text:
            summary_text = summary_text.split('## クラスタ別分析結果')[0]
        st.markdown(summary_text)
    except Exception as e:
        st.markdown(report.summary)  # エラーが発生した場合は全体を表示

    # クラスタ別詳細分析
    st.markdown("## クラスタ別詳細分析")
    cluster_sections = report.summary.split('### クラスタ')[1:]

    for i, section in enumerate(cluster_sections, 1):
        with st.expander(f"クラスタ {i} の詳細分析", expanded=True):
            # クラスタの基本情報
            cluster_content = section.strip()

            # クラスタ詳細セクション
            st.markdown("### クラスタ詳細")
            st.markdown("---")

            # 基本情報と概要
            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    compliance_status = "遵守" if "遵守状況: 遵守" in cluster_content else "未遵守"
                    
                    # スコアの抽出をより安全に行う
                    score = "N/A"
                    if "スコア: " in cluster_content:
                        score_parts = cluster_content.split("スコア: ")[1].split("\n")
                        if score_parts:
                            score = score_parts[0].strip()
                    
                    # 試行回数情報の抽出
                    trial_info = ""
                    if "判定結果（" in cluster_content and "が遵守と判定）" in cluster_content:
                        trial_text = cluster_content.split("判定結果（")[1].split("が遵守と判定）")[0]
                        if "/" in trial_text:
                            trial_info = trial_text
                    
                    st.metric("遵守状況", f"{compliance_status} ({trial_info})" if trial_info else compliance_status)
                    st.metric("平均スコア", score)
                except Exception as e:
                    print(f"Error parsing cluster content: {e}")
                    st.metric("遵守状況", "解析エラー")
                    st.metric("平均スコア", "N/A")


            with col2:
                summary = cluster_content.split("**要約:**\n")[1].split(
                    "\n**主要な発見事項:**")[0]
                st.markdown(summary)

            # 所属する要件・禁止事項
            st.markdown("#### 所属する要件・禁止事項")
            if "**対象となる主な要件:**" in cluster_content:
                requirements = cluster_content.split(
                    "**対象となる主な要件:**")[1].split("\n**主要な発見事項:**")[0]
                st.markdown(requirements)

            # 分析結果セクション
            st.markdown("### 分析結果")
            st.markdown("---")

            # 社内規定との整合性分析
            if "**分析コンテキスト:**" in cluster_content:
                st.markdown("#### 社内規定との整合性分析")
                analysis = cluster_content.split("**分析コンテキスト:**")[1].strip()
                st.markdown(analysis)

            # 主要な発見事項
            st.markdown("#### 主要な発見事項")
            findings = cluster_content.split("**主要な発見事項:**")[1].strip()
            for finding in findings.split("\n"):
                if finding.strip() and finding.startswith("-"):
                    st.markdown(finding)

            st.markdown("---")

    # 改善提案セクション
    if report.recommendations:
        st.markdown("## 改善提案")
        for i, cluster_recommendations in enumerate(report.recommendations, 1):
            with st.expander(f"クラスタ {i} の改善提案"):
                for recommendation in cluster_recommendations:
                    st.markdown(f"- {recommendation}")

    # Download Options
    st.markdown("## レポートのダウンロード")

    # JSON Download
    json_report = json.dumps(
        {
            'timestamp': report.timestamp.isoformat(),
            'total_requirements': report.total_requirements,
            'compliant_count': report.compliant_count,
            'non_compliant_count': report.non_compliant_count,
            'compliance_rate': f"{compliance_rate:.1f}%",
            'summary': report.summary,
            'recommendations': report.recommendations
        },
        ensure_ascii=False,
        indent=2)

    st.download_button(label="JSONレポートをダウンロード",
                       data=json_report,
                       file_name="compliance_report.json",
                       mime="application/json")

    # Markdown Download
    markdown_report = f"""# 分析レポート
生成日時: {report.timestamp}

## 分析概要
- 要件・禁止事項総数: {report.total_requirements}
- 遵守クラスタ数: {report.compliant_count}
- 未遵守クラスタ数: {report.non_compliant_count}
- 遵守率: {compliance_rate:.1f}%

## 詳細分析
{report.summary}

## 改善提案
{chr(10).join([f"- {item}" for sublist in report.recommendations for item in sublist])}
"""

    st.download_button(label="Markdownレポートをダウンロード",
                       data=markdown_report,
                       file_name="compliance_report.md",
                       mime="text/markdown")
    
    # PDF Download
    if st.button("PDFレポートを生成"):
        with st.spinner("PDFレポートを生成中..."):
            # Create PDF generator
            pdf_generator = PDFReportGenerator("compliance_report.pdf")
            
            # Add title
            pdf_generator.add_title("コンプライアンス分析レポート")
            
            # Add summary section
            pdf_generator.add_heading("分析概要")
            summary_table_data = [
                ["項目", "値"],
                ["要件・禁止事項総数", str(total_items)],
                ["遵守クラスタ数", str(report.compliant_count)],
                ["遵守率", f"{compliance_rate:.1f}%"]
            ]
            pdf_generator.add_table(summary_table_data)
            
            # Add compliance rate chart using ReportLab
            pdf_generator.add_pie_chart(
                data=[report.compliant_count, report.non_compliant_count],
                labels=['遵守', '未遵守']
            )
            
            # Add detailed analysis
            pdf_generator.add_heading("クラスタ別詳細分析")
            for i, section in enumerate(cluster_sections, 1):
                pdf_generator.add_heading(f"クラスタ {i}", level=2)
                
                # Add cluster details
                cluster_content = section.strip()
                if "**要約:**" in cluster_content:
                    summary = cluster_content.split("**要約:**\n")[1].split("\n**主要な発見事項:**")[0]
                    pdf_generator.add_paragraph(summary)
                
                # Add findings
                if "**主要な発見事項:**" in cluster_content:
                    findings = cluster_content.split("**主要な発見事項:**")[1].strip()
                    pdf_generator.add_paragraph("主要な発見事項:")
                    for finding in findings.split("\n"):
                        if finding.strip() and finding.startswith("-"):
                            pdf_generator.add_paragraph(f"  {finding}")
            
            # Generate PDF
            pdf_generator.generate()
            
            # Read the generated PDF file
            with open("compliance_report.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Create download button for PDF
            st.download_button(
                label="PDFレポートをダウンロード",
                data=pdf_bytes,
                file_name="compliance_report.pdf",
                mime="application/pdf"
            )
            
            # Clean up the temporary PDF file
            if os.path.exists("compliance_report.pdf"):
                os.remove("compliance_report.pdf")