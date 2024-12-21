import streamlit as st
from processors.gpt_processor import GPTProcessor
import json
import os
import base64
import io
from models.compliance_model import ComplianceReport
from datetime import datetime
import plotly.graph_objects as go
from utils.pdf_generator import PDFReportGenerator


def render_report_section():
    if not st.session_state.get('compliance_results'):
        st.info("先に分析を完了してください")
        return

    # クラスタ分析結果のファイル名を生成
    analysis_file = None
    if 'documents' in st.session_state:
        import hashlib
        legal_hash = hashlib.md5(st.session_state.documents['legal'].encode()).hexdigest()[:8]
        internal_hash = hashlib.md5(st.session_state.documents['internal'].encode()).hexdigest()[:8]
        analysis_file = f"data/cluster_analysis_{legal_hash}_{internal_hash}.json"

    # 既存の分析結果を確認
    import os
    if analysis_file and os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                import json
                st.session_state.compliance_results = json.load(f)
                st.success("既存の分析結果を読み込みました")
        except Exception as e:
            st.warning(f"分析結果の読み込みに失敗しました: {str(e)}")
            st.session_state.compliance_results = None
    
    # 分析結果がない場合は新規分析を実行
    if not st.session_state.get('compliance_results'):
        with st.spinner("クラスタ分析を実行中..."):
            results = analyze_compliance(
                st.session_state.analysis_results['legal']['requirements'],
                st.session_state.analysis_results['legal']['prohibitions'],
                st.session_state.embedding_processor
            )
            st.session_state.compliance_results = results
            
            # 分析結果を保存
            if analysis_file:
                try:
                    os.makedirs('data', exist_ok=True)
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    st.success("分析結果を保存しました")
                except Exception as e:
                    st.warning(f"分析結果の保存に失敗しました: {str(e)}")

    # レポート生成
    with st.spinner("レポートを生成中..."):
        st.session_state.generated_report = generate_compliance_report()
        
    display_report(st.session_state.generated_report)


def generate_compliance_report() -> ComplianceReport:
    """Generate comprehensive compliance report based on cluster analysis"""
    gpt_processor = GPTProcessor()

    if not st.session_state.get('compliance_results'):
        raise ValueError("分析結果が見つかりません")

    # クラスタ分析結果を取得
    clusters = st.session_state.compliance_results
    
    if not clusters:
        st.error("クラスタ分析結果が空です")
        return None

    # 基本メトリクスを計算
    total_clusters = len(clusters)
    compliant_clusters = sum(
        1 for cluster in clusters
        if cluster.get('analysis', {}).get('overall_compliance', False))

    # クラスタから要件と禁止事項の総数を計算
    total_requirements = sum(len(cluster.get('requirements', [])) for cluster in clusters)
    total_prohibitions = sum(len(cluster.get('prohibitions', [])) for cluster in clusters)
    print(f"Total requirements: {total_requirements}, Total prohibitions: {total_prohibitions}")

    # レポートのサマリーを生成
    print(f"Generating report for {len(clusters)} clusters")
    
    # メトリクス計算を最適化
    total_requirements = sum(len(cluster.get('requirements', [])) for cluster in clusters)
    total_prohibitions = sum(len(cluster.get('prohibitions', [])) for cluster in clusters)
    
    summary_parts = [
        "# クラスタベース分析レポート\n", 
        f"## 概要\n",
        f"以下の分析結果は、{len(clusters)}個のクラスタに基づいています：\n",
        f"- 分析対象の要件総数: {total_requirements}件",
        f"- 分析対象の禁止事項総数: {total_prohibitions}件",
        f"- 遵守クラスタ数: {compliant_clusters}個",
        f"- 全体遵守率: {(compliant_clusters / total_clusters * 100):.1f}% ({compliant_clusters}/{total_clusters})",
        "\nこの分析結果は、各要件と禁止事項を意味的に関連するグループ（クラスタ）に分類し、",
        "各クラスタごとの遵守状況を総合的に評価したものです。\n",
        "\n## クラスタ別分析結果\n"
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
    # 要件と禁止事項の合計を計算
    total_items = total_requirements + total_prohibitions
    
    report = ComplianceReport(
        timestamp=datetime.now(),
        legal_document_name="法令文書",
        internal_document_name="社内規定",
        total_requirements=total_items,  # 要件と禁止事項の合計を設定
        compliant_count=compliant_clusters,
        non_compliant_count=total_clusters - compliant_clusters,
        matches=[],  # 新しい構造では使用しない
        gaps=[],  # 新しい構造では使用しない
        summary=report_content,
        recommendations=[]  # 初期化時に空のリストを設定
    )

    # 現在のクラスタ数に基づいて改善提案を生成
    recommendations = []
    for cluster in clusters:
        if cluster.get('analysis') and cluster['analysis'].get('improvement_suggestions'):
            recommendations.append(cluster['analysis']['improvement_suggestions'])
        
    # クラスタ数と改善提案の数が一致することを確認
    if len(recommendations) != len(clusters):
        print(f"Warning: Number of recommendations ({len(recommendations)}) does not match number of clusters ({len(clusters)})")
        # 不足分は空のリストで埋める
        while len(recommendations) < len(clusters):
            recommendations.append([])
        # 余分な提案は削除
        recommendations = recommendations[:len(clusters)]
    
    report.recommendations = recommendations

    return report


def display_report(report: ComplianceReport):
    """Display the cluster-based compliance report in the UI"""
    st.header("評価結果レポート")
    st.markdown(f"**生成日時:** {report.timestamp.strftime('%Y年%m月%d日 %H:%M')}")

    # エグゼクティブサマリー
    st.markdown("## サマリー")
    col1, col2, col3 = st.columns(3)

    # 要件・禁止事項の総数を計算（要件数と禁止事項数の合計）
    total_items = sum(len(cluster.get('requirements', [])) for cluster in st.session_state.compliance_results) + \
                 sum(len(cluster.get('prohibitions', [])) for cluster in st.session_state.compliance_results)

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
    
    # compliance_resultsから直接クラスタ情報を取得
    if 'compliance_results' not in st.session_state:
        st.warning("クラスタ分析結果が見つかりません")
        return
        
    clusters = st.session_state.compliance_results
    print(f"Processing {len(clusters)} clusters from compliance_results")
    
    for cluster in clusters:
        cluster_id = cluster.get('cluster_id', 'Unknown')
        with st.expander(f"クラスタ {cluster_id} の詳細分析", expanded=True):
            # クラスタの基本情報
            analysis = cluster.get('analysis', {})
            summary = cluster.get('summary', {})

            # クラスタ詳細セクション
            st.markdown("### クラスタ詳細")
            st.markdown("---")

            # 基本情報と概要
            col1, col2 = st.columns([1, 2])
            with col1:
                try:
                    compliance_status = "遵守" if analysis.get('overall_compliance', False) else "未遵守"
                    compliance_score = analysis.get('compliance_score', 0.0)
                    
                    st.metric("遵守状況", compliance_status)
                    st.metric("コンプライアンススコア", f"{compliance_score:.2f}")
                except Exception as e:
                    print(f"Error displaying cluster metrics: {e}")
                    st.metric("遵守状況", "解析エラー")
                    st.metric("コンプライアンススコア", "N/A")

            with col2:
                if summary.get('comprehensive_summary'):
                    st.markdown(summary['comprehensive_summary'])
                else:
                    st.markdown("要約情報がありません")

            # 要件と禁止事項のリスト
            st.markdown("### 要件・禁止事項一覧")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 要件")
                for req in cluster['requirements']:
                    st.markdown(f"- {req['text']}")

            with col2:
                st.markdown("#### 禁止事項")
                for prob in cluster['prohibitions']:
                    st.markdown(f"- {prob['text']}")

            # 分析結果
            st.markdown("### 分析結果")
            st.markdown(f"**コンプライアンススコア:** {cluster['analysis']['compliance_score']:.2f}")
            st.write(cluster['analysis']['analysis'])

            # 発見事項と改善提案
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 主要な発見事項")
                for finding in cluster['analysis']['key_findings']:
                    st.markdown(f"- {finding}")

            with col2:
                st.markdown("#### 改善提案")
                for suggestion in cluster['analysis']['improvement_suggestions']:
                    st.markdown(f"- {suggestion}")

            st.markdown("---")

    # 改善提案セクション
    if report.recommendations and len(clusters) > 0:
        st.markdown("## 改善提案")
        # 現在のクラスタ数に基づいて改善提案を表示
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id', 'Unknown')
            # クラスタIDに対応する改善提案を取得
            cluster_recommendations = cluster.get('analysis', {}).get('improvement_suggestions', [])
            with st.expander(f"クラスタ {cluster_id} の改善提案"):
                if cluster_recommendations:
                    for recommendation in cluster_recommendations:
                        st.markdown(f"- {recommendation}")
                else:
                    st.markdown("このクラスタの改善提案はありません")

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

    st.download_button(
        label="JSONレポートをダウンロード",
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

    st.download_button(
        label="Markdownレポートをダウンロード",
        data=markdown_report,
        file_name="compliance_report.md",
        mime="text/markdown")
    
    # PDF自動生成と表示
    st.markdown("## PDFレポートプレビュー")
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
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id', 'Unknown')
            pdf_generator.add_heading(f"クラスタ {cluster_id}", level=2)
            
            # Add cluster summary
            if cluster.get('summary', {}).get('comprehensive_summary'):
                pdf_generator.add_paragraph(cluster['summary']['comprehensive_summary'])
            
            # Add findings
            if cluster.get('analysis', {}).get('key_findings'):
                pdf_generator.add_paragraph("主要な発見事項:")
                for finding in cluster['analysis']['key_findings']:
                    pdf_generator.add_paragraph(f"  - {finding}")
        
        # Generate PDF
        if not pdf_generator.generate():
            st.error("PDFの生成に失敗しました。詳細はログを確認してください。")
            return
            
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
        
        # Display PDF preview
        if os.path.exists("compliance_report.pdf"):
            with open("compliance_report.pdf", "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
                # PDFをiframeで表示
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}" width="100%" height="800" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                # PDFの内容を検証（文字化けチェック）
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text()
                    
                    # 文字化けの検出
                    if '�' in text_content or not any('\u4e00' <= c <= '\u9fff' for c in text_content):
                        st.error("警告: PDFで日本語の文字化けが検出されました。")
                        # エラーログの記録
                        print(f"PDF文字化けエラー: {text_content[:200]}...")
                    else:
                        st.success("PDFの日本語表示は正常です。")
                except Exception as e:
                    st.error(f"PDFの検証中にエラーが発生しました: {str(e)}")