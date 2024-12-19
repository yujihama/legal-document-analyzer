import streamlit as st
from processors.gpt_processor import GPTProcessor
import json
from models.compliance_model import ComplianceReport
from datetime import datetime

def render_report_section():
    headers = {
        'ja': "コンプライアンスレポート",
        'en': "Compliance Report"
    }
    info_messages = {
        'ja': "先に分析を完了してください",
        'en': "Please complete the analysis first"
    }
    
    st.header(headers[st.session_state.language])
    
    if not st.session_state.get('compliance_results'):
        st.info(info_messages[st.session_state.language])
        return
    
    if 'generated_report' not in st.session_state:
        with st.spinner("Generating comprehensive report..."):
            st.session_state.generated_report = generate_compliance_report()
    
    display_report(st.session_state.generated_report)

def generate_compliance_report() -> ComplianceReport:
    """Generate comprehensive compliance report based on cluster analysis"""
    gpt_processor = GPTProcessor()
    
    if not st.session_state.get('compliance_results'):
        raise ValueError("コンプライアンス分析結果が見つかりません")
    
    clusters = st.session_state.compliance_results
    
    # Count total requirements and compliant clusters
    total_clusters = len(clusters)
    compliant_clusters = sum(
        1 for cluster in clusters 
        if cluster.get('analysis', {}).get('overall_compliance', False)
    )
    
    # Calculate total requirements and prohibitions
    total_requirements = sum(len(cluster.get('requirements', [])) for cluster in clusters)
    total_prohibitions = sum(len(cluster.get('prohibitions', [])) for cluster in clusters)
    
    # Generate summary content
    summary_parts = [
        "# クラスタベースコンプライアンス分析レポート\n",
        f"## 概要\n",
        f"- 分析クラスタ数: {total_clusters}",
        f"- 要件総数: {total_requirements}",
        f"- 禁止事項総数: {total_prohibitions}",
        f"- 遵守クラスタ数: {compliant_clusters}",
        f"- 全体遵守率: {(compliant_clusters / total_clusters * 100):.1f}% ({compliant_clusters}/{total_clusters})",
        "\n## クラスタ別分析結果\n"
    ]
    
    # Add cluster-specific summaries
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', 'Unknown')
        analysis = cluster.get('analysis', {})
        summary = cluster.get('summary', {})
        
        summary_parts.extend([
            f"\n### クラスタ {cluster_id}",
            f"- 遵守状況: {'遵守' if analysis.get('overall_compliance', False) else '未遵守'}",
            f"- コンプライアンススコア: {analysis.get('compliance_score', 0.0):.2f}",
            f"\n**要約:**\n{summary.get('comprehensive_summary', '要約なし')}\n",
            "\n**主要な発見事項:**"
        ])
        
        # Add key findings
        for finding in analysis.get('key_findings', ['発見事項なし']):
            summary_parts.append(f"- {finding}")
    
    report_content = '\n'.join(summary_parts)
    
    # Create report object with updated structure
    report = ComplianceReport(
        timestamp=datetime.now(),
        legal_document_name="法令文書",
        internal_document_name="社内規定",
        total_requirements=total_requirements + total_prohibitions,
        compliant_count=compliant_clusters,
        non_compliant_count=total_clusters - compliant_clusters,
        matches=[],  # 新しい構造では使用しない
        gaps=[],     # 新しい構造では使用しない
        summary=report_content,
        recommendations=[cluster.get('analysis', {}).get('improvement_suggestions', [])
                        for cluster in clusters]
    )
    
    return report

def display_report(report: ComplianceReport):
    """Display the cluster-based compliance report in the UI"""
    # Summary Statistics
    st.markdown("## 分析概要")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("要件・禁止事項総数", report.total_requirements)
    with col2:
        st.metric("遵守クラスタ数", report.compliant_count)
    with col3:
        compliance_rate = (report.compliant_count / (report.compliant_count + report.non_compliant_count) * 100) \
            if (report.compliant_count + report.non_compliant_count) > 0 else 0
        st.metric("遵守率", f"{compliance_rate:.1f}%")
    
    # Report Content
    st.markdown("## 詳細分析")
    st.markdown(report.summary)
    
    # Recommendations if available
    if report.recommendations:
        st.markdown("## 改善提案")
        for cluster_recommendations in report.recommendations:
            for recommendation in cluster_recommendations:
                st.markdown(f"- {recommendation}")
    
    # Download Options
    st.markdown("## レポートのダウンロード")
    
    # JSON Download
    json_report = json.dumps({
        'timestamp': report.timestamp.isoformat(),
        'total_requirements': report.total_requirements,
        'compliant_count': report.compliant_count,
        'non_compliant_count': report.non_compliant_count,
        'compliance_rate': f"{compliance_rate:.1f}%",
        'summary': report.summary,
        'recommendations': report.recommendations
    }, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="JSONレポートをダウンロード",
        data=json_report,
        file_name="compliance_report.json",
        mime="application/json"
    )
    
    # Markdown Download
    markdown_report = f"""# コンプライアンス分析レポート
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
    
    st.download_button(
        label="Markdownレポートをダウンロード",
        data=markdown_report,
        file_name="compliance_report.md",
        mime="text/markdown"
    )
