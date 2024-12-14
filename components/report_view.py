import streamlit as st
from processors.gpt_processor import GPTProcessor
import json
from models.compliance_model import ComplianceReport
from datetime import datetime

def render_report_section():
    st.header("Compliance Report")
    
    if not st.session_state.get('compliance_results'):
        st.info("Please complete the analysis first")
        return
    
    if 'generated_report' not in st.session_state:
        with st.spinner("Generating comprehensive report..."):
            st.session_state.generated_report = generate_compliance_report()
    
    display_report(st.session_state.generated_report)

def generate_compliance_report() -> ComplianceReport:
    """Generate comprehensive compliance report"""
    gpt_processor = GPTProcessor()
    
    # Prepare analysis results for report generation
    analysis_data = {
        'compliance_results': st.session_state.compliance_results,
        'documents': {
            'legal': st.session_state.documents['legal'],
            'internal': st.session_state.documents['internal']
        }
    }
    
    # Generate report content using GPT
    report_content = gpt_processor.generate_report(analysis_data)
    
    # Count compliance statistics
    compliant_count = sum(
        1 for r in st.session_state.compliance_results 
        if any(m['analysis']['compliant'] for m in r['matches'])
    )
    total_count = len(st.session_state.compliance_results)
    
    # Create report object
    report = ComplianceReport(
        timestamp=datetime.now(),
        legal_document_name="Legal Document",
        internal_document_name="Internal Regulations",
        total_requirements=total_count,
        compliant_count=compliant_count,
        non_compliant_count=total_count - compliant_count,
        matches=[],  # Detailed matches would be populated here
        gaps=[],     # Gaps would be populated here
        summary=report_content,
        recommendations=[]  # Recommendations would be populated here
    )
    
    return report

def display_report(report: ComplianceReport):
    """Display the compliance report in the UI"""
    # Summary Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requirements", report.total_requirements)
    with col2:
        st.metric("Compliant Items", report.compliant_count)
    with col3:
        st.metric("Non-Compliant Items", report.non_compliant_count)
    
    # Report Content
    st.markdown("## Executive Summary")
    st.markdown(report.summary)
    
    # Download Options
    st.markdown("## Download Report")
    
    # JSON Download
    json_report = json.dumps(report.to_dict(), indent=2)
    st.download_button(
        label="Download JSON Report",
        data=json_report,
        file_name="compliance_report.json",
        mime="application/json"
    )
    
    # Markdown Download
    markdown_report = f"""# Compliance Analysis Report
Generated: {report.timestamp}

## Summary Statistics
- Total Requirements: {report.total_requirements}
- Compliant Items: {report.compliant_count}
- Non-Compliant Items: {report.non_compliant_count}

## Detailed Analysis
{report.summary}
"""
    
    st.download_button(
        label="Download Markdown Report",
        data=markdown_report,
        file_name="compliance_report.md",
        mime="text/markdown"
    )
