import streamlit as st
from components.document_upload import render_upload_section
from components.analysis_view import render_analysis_section
from components.report_view import render_report_section

st.set_page_config(
    page_title="Legal Compliance Checker",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("Legal Compliance Analysis System")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'documents' not in st.session_state:
        st.session_state.documents = {'legal': None, 'internal': None}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Navigation
    tabs = st.tabs(["Document Upload", "Analysis", "Report"])
    
    with tabs[0]:
        render_upload_section()
    
    with tabs[1]:
        render_analysis_section()
    
    with tabs[2]:
        render_report_section()

if __name__ == "__main__":
    main()
