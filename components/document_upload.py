import streamlit as st
from processors.document_processor import DocumentProcessor
import io

def load_sample_data():
    """Load sample data files if no documents are uploaded"""
    try:
        if 'documents' not in st.session_state:
            st.session_state.documents = {'legal': '', 'internal': ''}
        
        if not st.session_state.documents['legal'] or not st.session_state.documents['internal']:
            with open('sample_data/legal_document.txt', 'r', encoding='utf-8') as f:
                st.session_state.documents['legal'] = f.read()
            with open('sample_data/internal_document.txt', 'r', encoding='utf-8') as f:
                st.session_state.documents['internal'] = f.read()
            return True
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return False

def render_upload_section():
    st.header("Document Upload")
    
    # Load sample data if no documents are present
    if load_sample_data():
        st.info("サンプルデータが自動的にロードされました。必要に応じて新しい文書をアップロードしてください。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Legal Document")
        legal_file = st.file_uploader(
            "Upload legal document (PDF/TXT)",
            type=['pdf', 'txt'],
            key="legal_doc"
        )
        
        if legal_file:
            text = read_file(legal_file)
            st.session_state.documents['legal'] = text
            st.success("Legal document uploaded successfully")
    
    with col2:
        st.subheader("Internal Regulations")
        internal_file = st.file_uploader(
            "Upload internal regulations (PDF/TXT)",
            type=['pdf', 'txt'],
            key="internal_doc"
        )
        
        if internal_file:
            text = read_file(internal_file)
            st.session_state.documents['internal'] = text
            st.success("Internal regulations uploaded successfully")
    
    # Display processing results if available
    if 'analysis_results' in st.session_state and st.session_state.analysis_results:
        st.subheader("処理結果")
        
        # Display legal document results
        st.markdown("### 法令文書の解析結果")
        legal_results = st.session_state.analysis_results['legal']
        st.write(f"抽出されたセクション数: {len(legal_results['sections'])}")
        st.write(f"要求事項数: {len(legal_results['requirements'])}")
        st.write(f"禁止事項数: {len(legal_results['prohibitions'])}")
        
        # Display internal document results
        st.markdown("### 社内規定文書の解析結果")
        internal_results = st.session_state.analysis_results['internal']
        st.write(f"抽出されたセクション数: {len(internal_results['sections'])}")
        
        # Debug information display
        # if 'processing_results' in st.session_state:
        #     st.subheader("処理詳細")
            
        #     # Legal document processing details
        #     st.markdown("### 法令文書の処理詳細")
        #     for debug_info in st.session_state.processing_results['legal']['debug_info']:
        #         st.markdown(f"#### {debug_info['title']}")
        #         st.write("**Input:**")
        #         st.code(debug_info['input'])
        #         st.write("**Response:**")
        #         st.json(debug_info['response'])
        #         st.markdown("---")
            
        #     # Internal document processing details
        #     st.markdown("### 社内規定文書の処理詳細")
        #     for debug_info in st.session_state.processing_results['internal']['debug_info']:
        #         st.markdown(f"#### {debug_info['title']}")
        #         st.write("**Input:**")
        #         st.code(debug_info['input'])
        #         st.write("**Response:**")
        #         st.json(debug_info['response'])
        #         st.markdown("---")
    
    if st.session_state.documents['legal'] and st.session_state.documents['internal']:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Initialize processing results in session state if not exists
                if 'processing_results' not in st.session_state:
                    st.session_state.processing_results = {
                        'legal': {'debug_info': []},
                        'internal': {'debug_info': []}
                    }
                
                processor = DocumentProcessor(language=st.session_state.language)
                
                # Process legal document
                st.info("Step 1: Processing legal document...")
                legal_results = processor.process_legal_document(
                    st.session_state.documents['legal']
                )
                st.success("Legal document processed")
                
                # Process internal document
                st.info("Step 2: Processing internal document...")
                internal_results = processor.process_internal_document(
                    st.session_state.documents['internal']
                )
                st.success("Internal document processed")
                
                st.info("Step 3: Combining results...")
                st.session_state.analysis_results = {
                    'legal': legal_results,
                    'internal': internal_results
                }
                
                st.session_state.current_step = 'analysis'
                st.success("全ての文書の処理が完了しました！")
                st.write("「Analysis」タブに移動して、詳細な分析結果を確認してください。")
                st.rerun()

def read_file(uploaded_file) -> str:
    """Read uploaded file and return text content"""
    if uploaded_file.type == "application/pdf":
        try:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    else:
        return uploaded_file.getvalue().decode('utf-8')
