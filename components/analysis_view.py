import streamlit as st
import plotly.graph_objects as go
from processors.embedding_processor import EmbeddingProcessor
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
        st.session_state.compliance_results = analyze_compliance(
            results['legal']['requirements'],
            results['legal']['prohibitions'],
            st.session_state.embedding_processor
        )
    
    display_compliance_results(st.session_state.compliance_results)

def analyze_compliance(requirements, prohibitions, embedding_processor):
    """Analyze compliance for requirements and prohibitions"""
    gpt_processor = GPTProcessor()
    results = []
    
    for req in requirements + prohibitions:
        # Find similar sections in internal regulations
        similar = embedding_processor.find_similar(req['text'], k=3)
        
        # Analyze compliance for each similar section
        compliance_status = []
        for match in similar:
            analysis = gpt_processor.analyze_compliance(req['text'], match['text'])
            compliance_status.append({
                'text': match['text'],
                'analysis': analysis,
                'similarity_score': match['score']
            })
        
        results.append({
            'requirement': req,
            'matches': compliance_status
        })
    
    return results

def display_compliance_results(results):
    """Display compliance analysis results with visualizations"""
    # Compliance Status Chart
    compliant = sum(1 for r in results if any(
        m['analysis']['compliant'] for m in r['matches']
    ))
    non_compliant = len(results) - compliant
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Compliant', 'Non-Compliant'],
            values=[compliant, non_compliant],
            marker_colors=['#2ecc71', '#e74c3c']
        )
    ])
    fig.update_layout(title="Compliance Status Overview")
    st.plotly_chart(fig)
    
    # Detailed Results
    st.subheader("Detailed Analysis")
    for result in results:
        with st.expander(f"Requirement: {result['requirement']['text'][:100]}..."):
            st.write("**Requirement Type:**", 
                    "Prohibition" if result['requirement'].get('is_prohibition') 
                    else "Requirement")
            st.write("**Matches in Internal Regulations:**")
            
            for match in result['matches']:
                st.markdown("---")
                st.write("**Matched Text:**", match['text'])
                st.write("**Analysis:**", match['analysis']['explanation'])
                st.write("**Compliance Score:**", 
                        f"{match['analysis'].get('score', 0):.2f}")
