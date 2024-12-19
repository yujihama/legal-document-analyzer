import streamlit as st
import plotly.graph_objects as go
import numpy as np
from processors.embedding_processor import EmbeddingProcessor, ClusterInfo
from processors.gpt_processor import GPTProcessor

def render_analysis_section():
    st.header("Compliance Analysis")
    
    if not st.session_state.analysis_results:
        st.info("Please upload and process documents first")
        return
    
    results = st.session_state.analysis_results
    
    # Display Cluster Analysis Summary
    st.subheader("クラスタ分析の概要")
    
    if 'compliance_results' in st.session_state:
        clusters = st.session_state.compliance_results
        
        col1, col2, col3 = st.columns(3)
        
        total_clusters = len(clusters)
        total_reqs = sum(len(cluster['requirements']) for cluster in clusters)
        total_prohibs = sum(len(cluster['prohibitions']) for cluster in clusters)
        
        with col1:
            st.metric("クラスタ数", total_clusters)
        with col2:
            st.metric("要件数", total_reqs)
        with col3:
            st.metric("禁止事項数", total_prohibs)
        
        # Display cluster analysis results
        st.subheader("クラスタ別分析結果")
    
    if 'embedding_processor' not in st.session_state:
        with st.spinner("Initializing analysis..."):
            processor = EmbeddingProcessor()
            internal_sections = [s['text'] for s in results['internal']['sections']]
            processor.create_index(internal_sections)
            st.session_state.embedding_processor = processor
    
    # Analyze compliance for each requirement
    if 'compliance_results' not in st.session_state:
        with st.spinner("分析を実行中..."):
            compliance_results = analyze_compliance(
                results['legal']['requirements'],
                results['legal']['prohibitions'],
                st.session_state.embedding_processor
            )
            st.session_state.compliance_results = compliance_results
            st.success("分析が完了しました")
    
    # Display results
    headers = {
        'ja': "遵守状況の概要",
        'en': "Compliance Status Overview"
    }
    st.subheader(headers[st.session_state.language])
    display_compliance_results(st.session_state.compliance_results)
    
    # Display detailed analysis
    headers_detail = {
        'ja': "詳細分析",
        'en': "Detailed Analysis"
    }
    labels = {
        'ja': {
            'requirement': "要件",
            'requirement_type': "要件タイプ",
            'prohibition': "禁止事項",
            'regular': "要求事項",
            'matches': "社内規定との一致",
            'matched_text': "一致したテキスト",
            'analysis': "分析結果",
            'score': "一致スコア"
        },
        'en': {
            'requirement': "Requirement",
            'requirement_type': "Requirement Type",
            'prohibition': "Prohibition",
            'regular': "Requirement",
            'matches': "Matches in Internal Regulations",
            'matched_text': "Matched Text",
            'analysis': "Analysis",
            'score': "Match Score"
        }
    }
    
    if 'compliance_results' in st.session_state:
        clusters = st.session_state.compliance_results
        for i, cluster in enumerate(clusters):
            with st.expander(f"クラスタ {cluster['cluster_id']} の分析結果"):
                # Display cluster summary
                st.markdown("### クラスタの要約")
                st.write(cluster['summary']['comprehensive_summary'])
                
                # Display requirements and prohibitions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 要件")
                    for req in cluster['requirements']:
                        st.markdown(f"- {req['text']}")
                
                with col2:
                    st.markdown("### 禁止事項")
                    for prob in cluster['prohibitions']:
                        st.markdown(f"- {prob['text']}")
                
                # Display compliance analysis
                st.markdown("### コンプライアンス分析")
                st.markdown(f"**遵守状況:** {'遵守' if cluster['analysis']['overall_compliance'] else '未遵守'}")
                st.markdown(f"**スコア:** {cluster['analysis']['compliance_score']:.2f}")
                st.markdown("**分析結果:**")
                st.write(cluster['analysis']['analysis'])
                
                # Display findings and suggestions
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 主要な発見事項")
                    for finding in cluster['analysis']['key_findings']:
                        st.markdown(f"- {finding}")
                
                with col2:
                    st.markdown("### 改善提案")
                    for suggestion in cluster['analysis']['improvement_suggestions']:
                        st.markdown(f"- {suggestion}")

import multiprocessing
from functools import partial

def process_requirement(args):
    """Process a single requirement in parallel"""
    try:
        req, stored_texts, cluster_data = args
        
        # Initialize processors in the child process
        embedding_processor = EmbeddingProcessor()
        embedding_processor.stored_texts = stored_texts
        embedding_processor.create_index(stored_texts)  # Recreate index
        gpt_processor = GPTProcessor()
        
        # Recreate clusters from serialized data
        clusters = []
        for c_data in cluster_data:
            try:
                centroid = np.array(c_data['centroid']) if c_data.get('centroid') else None
                cluster = ClusterInfo(
                    id=int(c_data['id']),
                    texts=c_data['texts'],
                    centroid=centroid,
                    representative_text=c_data.get('representative_text'),
                    summary=c_data.get('summary')
                )
                clusters.append(cluster)
            except Exception as e:
                print(f"Error recreating cluster: {e}")
                continue
        
        # Find similar sections in internal regulations
        try:
            similar = embedding_processor.find_similar(req['text'], k=3)
        except Exception as e:
            print(f"Error finding similar sections: {e}")
            similar = []
        
        # Find the most relevant cluster
        try:
            query_embedding = embedding_processor.get_embedding(req['text'])
            distances = []
            for cluster in clusters:
                if cluster.centroid is not None:
                    try:
                        distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                        distances.append((distance, cluster))
                    except Exception as e:
                        print(f"Error calculating distance for cluster {cluster.id}: {e}")
                        continue
            
            closest_cluster = min(distances, key=lambda x: x[0])[1] if distances else (clusters[0] if clusters else None)
        except Exception as e:
            print(f"Error finding closest cluster: {e}")
            closest_cluster = clusters[0] if clusters else None
        
        # Analyze compliance for each similar section
        compliance_status = []
        for match in similar:
            analysis = gpt_processor.analyze_compliance(req['text'], match['text'])
            compliance_status.append({
                'text': match['text'],
                'analysis': analysis,
                'similarity_score': match['score'],
                'cluster_info': closest_cluster.to_dict() if closest_cluster else {}
            })
        
        return {
            'requirement': req,
            'matches': compliance_status,
            'cluster': closest_cluster.to_dict() if closest_cluster else {}
        }
    except Exception as e:
        print(f"Error processing requirement: {e}")
        return None

def analyze_compliance(requirements, prohibitions, embedding_processor):
    """Analyze compliance using cluster-based approach"""
    try:
        if not requirements and not prohibitions:
            st.warning("要件または禁止事項が見つかりません")
            return []
            
        if not embedding_processor or not embedding_processor.stored_texts:
            st.warning("内部規定が見つかりません")
            return []
            
        gpt_processor = GPTProcessor()
        
        # First, perform clustering on internal regulations
        with st.spinner("クラスタリングを実行中..."):
            n_clusters = min(5, max(1, len(embedding_processor.stored_texts)))
            clusters = embedding_processor.perform_clustering(n_clusters=n_clusters)
            
            if not clusters:
                st.warning("クラスタリングが正常に実行できませんでした")
                return []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_clusters = len(clusters)
        
        results = []
        
        # Process each cluster
        for i, cluster in enumerate(clusters):
            try:
                # Find requirements and prohibitions belonging to this cluster
                cluster_reqs = []
                cluster_prohibs = []
                
                # Get query embedding for each requirement and prohibition
                for req in requirements:
                    query_embedding = embedding_processor.get_embedding(req['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    if distance < 1.5:  # Threshold for cluster membership
                        cluster_reqs.append(req)
                
                for prob in prohibitions:
                    query_embedding = embedding_processor.get_embedding(prob['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    if distance < 1.5:  # Threshold for cluster membership
                        cluster_prohibs.append(prob)
                
                # Generate comprehensive summary for the cluster
                cluster_summary = gpt_processor.summarize_cluster_requirements(
                    cluster_reqs, cluster_prohibs
                )
                
                # Get relevant internal regulations for this cluster
                cluster_regulations = []
                for text in cluster.texts:
                    if len(text.strip()) > 0:
                        cluster_regulations.append(text)
                
                # Analyze compliance for the cluster
                compliance_analysis = gpt_processor.analyze_cluster_compliance(
                    cluster_summary['comprehensive_summary'],
                    cluster_regulations
                )
                
                # Store results
                results.append({
                    'cluster_id': cluster.id,
                    'requirements': cluster_reqs,
                    'prohibitions': cluster_prohibs,
                    'summary': cluster_summary,
                    'analysis': compliance_analysis,
                    'regulations': cluster_regulations
                })
                
                # Update progress
                progress = (i + 1) / total_clusters
                progress_bar.progress(progress)
                status_text.text(f"クラスタ分析進捗: {i + 1}/{total_clusters}")
                
            except Exception as e:
                st.error(f"クラスタ {cluster.id} の処理中にエラーが発生しました: {str(e)}")
                print(f"Cluster processing error details: {e}")
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    except Exception as e:
        st.error(f"コンプライアンス分析中にエラーが発生しました: {str(e)}")
        return []

def display_compliance_results(results):
    """Display cluster-based compliance analysis results with visualizations"""
    if not results:
        st.warning("分析結果がありません")
        return
    
    # Validate result structure
    def is_valid_result(r):
        return (isinstance(r, dict) and
                'analysis' in r and
                isinstance(r['analysis'], dict) and
                'overall_compliance' in r['analysis'])
    
    # Filter valid results and count compliant clusters
    valid_results = [r for r in results if is_valid_result(r)]
    if not valid_results:
        st.error("有効な分析結果が見つかりませんでした")
        return
    
    # Overall compliance visualization
    compliant_clusters = sum(1 for r in valid_results if r['analysis']['overall_compliance'])
    total_clusters = len(valid_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Compliance Status Pie Chart
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=['遵守', '未遵守'],
                values=[compliant_clusters, total_clusters - compliant_clusters],
                marker_colors=['#2ecc71', '#e74c3c']
            )
        ])
        fig_pie.update_layout(title="クラスタごとの遵守状況")
        st.plotly_chart(fig_pie)
    
    with col2:
        # Compliance Scores Bar Chart
        fig_scores = go.Figure(data=[
            go.Bar(
                x=[f"クラスタ {r['cluster_id']}" for r in results],
                y=[r['analysis']['compliance_score'] for r in results],
                marker_color='#3498db'
            )
        ])
        fig_scores.update_layout(
            title="コンプライアンススコア分布",
            yaxis=dict(range=[0, 1]),
            yaxis_title="スコア"
        )
        st.plotly_chart(fig_scores)
    
    # Display detailed analysis for each cluster
    st.subheader("クラスタ別詳細分析")
    
    for result in results:
        with st.expander(f"クラスタ {result['cluster_id']} の詳細分析"):
            # Cluster summary
            st.markdown("### クラスタの要約")
            st.markdown("**包括的な要約:**")
            st.write(result['summary']['comprehensive_summary'])
            
            st.markdown("**重要なポイント:**")
            for point in result['summary']['key_points']:
                st.markdown(f"- {point}")
            
            # Requirements and prohibitions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 要件のまとめ")
                st.write(result['summary']['requirements_summary'])
                st.markdown("**個別要件:**")
                for req in result['requirements']:
                    st.markdown(f"- {req['text']}")
            
            with col2:
                st.markdown("### 禁止事項のまとめ")
                st.write(result['summary']['prohibitions_summary'])
                st.markdown("**個別禁止事項:**")
                for prob in result['prohibitions']:
                    st.markdown(f"- {prob['text']}")
            
            # Compliance analysis
            st.markdown("### コンプライアンス分析")
            st.markdown(f"**全体評価:** {'遵守' if result['analysis']['overall_compliance'] else '未遵守'}")
            st.markdown(f"**スコア:** {result['analysis']['compliance_score']:.2f}")
            st.markdown("**詳細分析:**")
            st.write(result['analysis']['analysis'])
            
            # Key findings and suggestions
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 主要な発見事項")
                for finding in result['analysis']['key_findings']:
                    st.markdown(f"- {finding}")
            
            with col2:
                st.markdown("### 改善提案")
                for suggestion in result['analysis']['improvement_suggestions']:
                    st.markdown(f"- {suggestion}")
            
            # Related regulations
            st.markdown("### 関連する社内規定")
            st.markdown("---")
            for i, reg in enumerate(result['regulations'], 1):
                st.markdown(f"**規定 {i}:**")
                st.write(reg)
                if i < len(result['regulations']):
                    st.markdown("---")
