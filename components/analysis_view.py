import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Optional
from processors.embedding_processor import EmbeddingProcessor, ClusterInfo
from processors.gpt_processor import GPTProcessor
from processors.clustering_processor import ClusteringProcessor
import hashlib
import json
from utils.persistence import load_processing_results, save_processing_results
import os

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
        # 元の抽出結果から直接カウント
        total_reqs = len(results['legal']['requirements'])
        total_prohibs = len(results['legal']['prohibitions'])

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
            # 法令文書と社内規定の両方からテキストを収集
            all_texts = []

            # 法令文書から要件と禁止事項を収集
            for req in results['legal']['requirements']:
                all_texts.append(req['text'])
            for prob in results['legal']['prohibitions']:
                all_texts.append(prob['text'])

            # 社内規定のチャンクを追加
            internal_chunks = results['internal']['chunks']
            all_texts.extend(internal_chunks)

            print(f"Total texts for clustering: {len(all_texts)}")
            processor.create_index(all_texts)
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
    st.subheader("遵守状況の概要")
    display_compliance_results(st.session_state.compliance_results)

    # Display cluster analysis results
    if 'compliance_results' in st.session_state:
        clusters = st.session_state.compliance_results
        for i, cluster in enumerate(clusters):
            with st.expander(f"クラスタ {cluster['cluster_id']} の分析結果", expanded=True):
                # 基本情報（メトリクス）
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("このクラスタの要件数", len(cluster['requirements']))
                with col2:
                    st.metric("このクラスタの禁止事項数", len(cluster['prohibitions']))
                with col3:
                    compliance_status = "遵守" if cluster['analysis']['overall_compliance'] else "未遵守"
                    st.metric("遵守状況", compliance_status)

                # クラスタの概要
                st.markdown("### クラスタの概要")
                st.markdown(cluster['summary']['comprehensive_summary'])

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
            similar = embedding_processor.find_similar(req['text'], distance_threshold=1.5)
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

def analyze_compliance(requirements, prohibitions, processor: EmbeddingProcessor) -> List[Dict]:
    """Perform compliance analysis across all requirements"""
    try:
        # Generate cache key based on requirements and prohibitions
        import hashlib
        import json
        from utils.persistence import load_processing_results, save_processing_results

        cache_data = {
            'requirements': [r['text'] for r in requirements],
            'prohibitions': [p['text'] for p in prohibitions]
        }
        # Generate a unique key based on the content of both documents
        if 'documents' in st.session_state:
            legal_hash = hashlib.md5(st.session_state.documents['legal'].encode()).hexdigest()[:8]
            internal_hash = hashlib.md5(st.session_state.documents['internal'].encode()).hexdigest()[:8]
            cache_file = f"cluster_analysis_{legal_hash}_{internal_hash}.json"
        else:
            cache_key = hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
            cache_file = f"cluster_analysis_{cache_key}.json"

        # Try to load cached results
        cached_results = load_processing_results(cache_file)
        if cached_results:
            print(f"Loading cached cluster analysis results from {cache_file}")
            print(f"Cached clusters count: {len(cached_results)}")
            return cached_results
        else:
            print("No cached results found, performing new clustering")

        gpt_processor = GPTProcessor()
        # Generate clusters
        print("Starting clustering process...")
        clusters = processor.perform_clustering(min_cluster_size=2)
        print(f"Generated clusters count: {len(clusters)}")

        if not clusters:
            st.error("クラスタが見つかりません")
            return []

        # Debug information
        print("Cluster details:")
        for cluster in clusters:
            print(f"Cluster ID: {cluster.id}, Texts count: {len(cluster.texts)}")

        total_clusters = len(clusters)
        results = []

        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()


        # Process each cluster
        for i, cluster in enumerate(clusters):
            try:
                # Find requirements and prohibitions belonging to this cluster
                cluster_reqs = []
                cluster_prohibs = []

                # Calculate distances for all requirements to this cluster
                req_distances = []
                for req in requirements:
                    query_embedding = processor.get_embedding(req['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    req_distances.append((distance, req))

                # Calculate distances for all prohibitions to this cluster
                prob_distances = []
                for prob in prohibitions:
                    query_embedding = processor.get_embedding(prob['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    prob_distances.append((distance, prob))

                # Sort by distance
                req_distances.sort(key=lambda x: x[0])
                prob_distances.sort(key=lambda x: x[0])

                # Assign requirements and prohibitions to this cluster if it's their closest cluster
                for dist, req in req_distances:
                    # Check if this cluster is the closest for this requirement
                    is_closest = True
                    for other_cluster in clusters:
                        if other_cluster.id != cluster.id:
                            other_embedding = processor.get_embedding(req['text'])
                            other_distance = float(np.linalg.norm(other_embedding - other_cluster.centroid))
                            if other_distance < dist:
                                is_closest = False
                                break
                    if is_closest:
                        cluster_reqs.append(req)

                for dist, prob in prob_distances:
                    # Check if this cluster is the closest for this prohibition
                    is_closest = True
                    for other_cluster in clusters:
                        if other_cluster.id != cluster.id:
                            other_embedding = processor.get_embedding(prob['text'])
                            other_distance = float(np.linalg.norm(other_embedding - other_cluster.centroid))
                            if other_distance < dist:
                                is_closest = False
                                break
                    if is_closest:
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

        # Save results to cache
        try:
            os.makedirs('data', exist_ok=True)
            print(f"Saving results to cache file: {cache_file}")
            print(f"Results data type: {type(results)}")
            print(f"Results length: {len(results)}")
            
            # Convert results to serializable format if needed
            serializable_results = []
            for cluster in results:
                if isinstance(cluster, dict):
                    serializable_results.append(cluster)
                else:
                    serializable_results.append(cluster.to_dict())
            
            save_processing_results(serializable_results, cache_file)
            print(f"Successfully saved results to cache file: {cache_file}")
            
            # Verify the file was created
            if os.path.exists(cache_file):
                print(f"Verified: Cache file exists at {cache_file}")
                print(f"File size: {os.path.getsize(cache_file)} bytes")
            else:
                print(f"Warning: Cache file was not created at {cache_file}")
                
        except Exception as e:
            print(f"Error saving results to cache: {str(e)}")
            st.error(f"キャッシュの保存中にエラーが発生しました: {str(e)}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
        return results
    except Exception as e:
        st.error(f"コンプライアンス分析中にエラーが発生しました: {str(e)}")
        return []

def display_compliance_results(clusters):
    """Display compliance analysis results with visualizations"""
    if not clusters:
        st.warning("クラスタ分析結果が見つかりません")
        return

    # Calculate metrics for visualization
    total_clusters = len(clusters)
    compliant_clusters = sum(1 for c in clusters if c['analysis']['overall_compliance'])
    non_compliant_clusters = total_clusters - compliant_clusters

    # Create pie chart for compliance status
    fig_pie = go.Figure(data=[go.Pie(
        labels=['遵守', '未遵守'],
        values=[compliant_clusters, non_compliant_clusters],
        hole=.3,
        marker_colors=['#00CC96', '#EF553B']
    )])
    fig_pie.update_layout(
        title='コンプライアンス状況の概要',
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig_pie)

    # Create bar chart for requirements and prohibitions by cluster
    cluster_ids = [str(c['cluster_id']) for c in clusters]
    req_counts = [len(c['requirements']) for c in clusters]
    prob_counts = [len(c['prohibitions']) for c in clusters]

    fig_bar = go.Figure(data=[
        go.Bar(name='要件', x=cluster_ids, y=req_counts),
        go.Bar(name='禁止事項', x=cluster_ids, y=prob_counts)
    ])
    fig_bar.update_layout(
        title='クラスタごとの要件・禁止事項数',
        xaxis_title='クラスタID',
        yaxis_title='数',
        barmode='group',
        height=400
    )
    st.plotly_chart(fig_bar)