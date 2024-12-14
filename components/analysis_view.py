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
    
    # Display Requirements Summary
    st.subheader("Requirements Summary")
    col1, col2, col3 = st.columns(3)
    
    total_reqs = len(results['legal']['requirements'])
    total_prohibs = len(results['legal']['prohibitions'])
    
    with col1:
        st.metric("Total Requirements", total_reqs)
    with col2:
        st.metric("Total Prohibitions", total_prohibs)
    with col3:
        st.metric("Total Rules", total_reqs + total_prohibs)
    
    # Requirements Analysis
    st.subheader("Requirements Analysis")
    
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
    
    st.subheader(headers_detail[st.session_state.language])
    for i, result in enumerate(st.session_state.compliance_results):
        with st.expander(f"{labels[st.session_state.language]['requirement']} {i+1}: {result['requirement']['text'][:100]}..."):
            st.markdown(f"**{labels[st.session_state.language]['requirement_type']}:**")
            st.write(labels[st.session_state.language]['prohibition'] if result['requirement'].get('is_prohibition') else labels[st.session_state.language]['regular'])
            
            st.markdown(f"**{labels[st.session_state.language]['matches']}:**")
            for match in result['matches']:
                st.markdown("---")
                st.markdown(f"**{labels[st.session_state.language]['matched_text']}:**")
                st.write(match['text'])
                st.markdown(f"**{labels[st.session_state.language]['analysis']}:**")
                st.write(match['analysis']['explanation'])
                st.markdown(f"**{labels[st.session_state.language]['score']}:**")
                st.write(f"{match['analysis'].get('score', 0):.2f}")

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
    """Analyze compliance for requirements and prohibitions using parallel processing"""
    try:
        gpt_processor = GPTProcessor()
        
        # First, perform clustering on internal regulations
        with st.spinner("クラスタリングを実行中..."):
            clusters = embedding_processor.perform_clustering(
                n_clusters=min(5, len(embedding_processor.stored_texts))
            )
            embedding_processor.update_cluster_representatives(gpt_processor)
        
        # Prepare serializable data for multiprocessing
        stored_texts = embedding_processor.stored_texts
        cluster_data = [cluster.to_dict() for cluster in clusters]
        
        # Prepare all requirements for processing
        all_reqs = requirements + prohibitions
        total_reqs = len(all_reqs)
        
        # Create args for each requirement
        process_args = [(req, stored_texts, cluster_data) for req in all_reqs]
        
        # Use a fixed small number of processes and fixed chunk size for stability
        num_processes = 2  # Limit to 2 processes
        chunk_size = 2     # Process 2 requirements at a time
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process requirements in parallel with limited processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = []
            try:
                # Use map instead of imap_unordered for better stability
                completed_results = pool.map(process_requirement, process_args, chunksize=chunk_size)
                
                # Process results sequentially after parallel computation
                for i, result in enumerate(completed_results):
                    if result:  # Only append valid results
                        results.append(result)
                        # Update progress
                        progress = (i + 1) / total_reqs
                        progress_bar.progress(progress)
                        status_text.text(f"分析進捗: {i + 1}/{total_reqs} 要件を処理完了")
            except Exception as e:
                st.error(f"並列処理中にエラーが発生しました: {str(e)}")
                print(f"Parallel processing error details: {e}")
                results = []  # Reset results on error
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    except Exception as e:
        st.error(f"コンプライアンス分析中にエラーが発生しました: {str(e)}")
        return []

def display_compliance_results(results):
    """Display compliance analysis results with visualizations"""
    # Compliance Status Chart
    compliant = sum(1 for r in results if any(
        m['analysis']['compliant'] for m in r['matches']
    ))
    non_compliant = len(results) - compliant
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Compliance Status Pie Chart
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=['遵守', '未遵守'],
                values=[compliant, non_compliant],
                marker_colors=['#2ecc71', '#e74c3c']
            )
        ])
        fig_pie.update_layout(title="遵守状況の概要")
        st.plotly_chart(fig_pie)
    
    with col2:
        # Display Cluster Information
        st.subheader("クラスタリング分析結果")
        if any(r.get('cluster') for r in results):
            clusters = {}
            for result in results:
                cluster = result.get('cluster', {})
                cluster_id = cluster.get('id')
                if cluster_id is not None:
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            'count': 0,
                            'summary': cluster.get('summary', ''),
                            'representative_text': cluster.get('representative_text', '')
                        }
                    clusters[cluster_id]['count'] += 1
            
            # Create cluster distribution chart
            cluster_counts = [{'id': k, 'count': v['count']} for k, v in clusters.items()]
            if cluster_counts:
                fig_cluster = go.Figure(data=[
                    go.Bar(
                        x=[f"クラスタ {c['id']}" for c in cluster_counts],
                        y=[c['count'] for c in cluster_counts],
                        marker_color='#3498db'
                    )
                ])
                fig_cluster.update_layout(
                    title="要件のクラスタ分布",
                    xaxis_title="クラスタ",
                    yaxis_title="要件数"
                )
                st.plotly_chart(fig_cluster)
    
    # Display detailed cluster information
    st.subheader("クラスタの詳細情報")
    if any(r.get('cluster') for r in results):
        for cluster_id, cluster_info in clusters.items():
            with st.expander(f"クラスタ {cluster_id} の詳細"):
                st.markdown("**代表的なテキスト:**")
                st.write(cluster_info['representative_text'])
                st.markdown("**クラスタの要約:**")
                st.write(cluster_info['summary'])
                st.markdown(f"**このクラスタに含まれる要件数:** {cluster_info['count']}")
                st.markdown("**クラスタ内のテキスト一覧:**")
                for i, text in enumerate(cluster_info.get('texts', []), 1):
                    st.write(f"{i}. {text[:30]}...")
