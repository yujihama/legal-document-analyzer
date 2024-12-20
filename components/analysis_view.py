import streamlit as st
import plotly.graph_objects as go
import numpy as np
from processors.embedding_processor import EmbeddingProcessor, ClusterInfo
from processors.gpt_processor import GPTProcessor
from processors.clustering_processor import ClusteringProcessor

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
    headers = {
        'ja': "遵守状況の概要",
        'en': "Compliance Status Overview"
    }
    st.subheader(headers[st.session_state.language])
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

        # デフォルトのクラスタリングパラメータを設定
        params = {
            'min_cluster_size': 2,
            'min_samples': 1,
            'cluster_selection_epsilon': 0.2
        }
        selected_method = 'hdbscan'

        # First, perform clustering on internal regulations
        with st.spinner("クラスタリングを実行中..."):
            try:
                n_texts = len(embedding_processor.stored_texts)
                if n_texts == 0:
                    st.warning("分析対象のテキストが見つかりません")
                    return []

                # データ数に基づいてパラメータを調整
                n_samples = len(embedding_processor.stored_texts)
                adjusted_params = params.copy()

                # 各メソッドに応じたパラメータの調整
                if selected_method == 'hierarchical':
                    adjusted_params['max_clusters'] = min(
                        adjusted_params.get('max_clusters', 10),
                        max(2, n_samples - 1)
                    )
                elif selected_method == 'dpmm':
                    adjusted_params['max_components'] = min(
                        adjusted_params.get('max_components', 10),
                        max(2, n_samples - 1)
                    )
                elif selected_method == 'hdbscan':
                    adjusted_params['min_cluster_size'] = min(
                        adjusted_params.get('min_cluster_size', 2),
                        max(2, n_samples // 2)
                    )

                # クラスタリングの実行
                clustering_processor = ClusteringProcessor(method_name=selected_method)
                embeddings = embedding_processor.batch_embed_texts(embedding_processor.stored_texts)

                print(f"Clustering with method: {selected_method}")
                print(f"Number of samples: {n_samples}")
                print(f"Adjusted parameters: {adjusted_params}")

                clusters = clustering_processor.perform_clustering(
                    embeddings=embeddings,
                    texts=embedding_processor.stored_texts,
                    **adjusted_params
                )

                if not clusters:
                    st.warning("クラスタリングが正常に実行できませんでした")
                    return []

                st.success(f"{len(clusters)}個のクラスタを生成しました")

            except Exception as e:
                st.error(f"クラスタリング中にエラーが発生しました: {str(e)}")
                print(f"Clustering error details: {str(e)}")  # デバッグ用
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

                # Calculate distances for all requirements to this cluster
                req_distances = []
                for req in requirements:
                    query_embedding = embedding_processor.get_embedding(req['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    req_distances.append((distance, req))

                # Calculate distances for all prohibitions to this cluster
                prob_distances = []
                for prob in prohibitions:
                    query_embedding = embedding_processor.get_embedding(prob['text'])
                    distance = float(np.linalg.norm(query_embedding - cluster.centroid))
                    prob_distances.append((distance, prob))

                # Sort by distance and get the closest items
                req_distances.sort(key=lambda x: x[0])
                prob_distances.sort(key=lambda x: x[0])

                # Select items that are closest to this cluster (top N items)
                cluster_size = max(2, len(requirements) // total_clusters)
                cluster_reqs = [req for _, req in req_distances[:cluster_size] if _ < 1.5]
                cluster_prohibs = [prob for _, prob in prob_distances[:cluster_size] if _ < 1.5]

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