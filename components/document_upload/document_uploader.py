import streamlit as st
from typing import Optional
from pathlib import Path
import tempfile

def render_upload_section():
    """文書アップロードセクションのレンダリング"""
    st.header("文書アップロード")
    
    # 法的文書のアップロード
    legal_doc = st.file_uploader(
        "法的文書をアップロード",
        type=["pdf", "txt", "docx"],
        key="legal_document"
    )
    
    # 内部文書のアップロード
    internal_doc = st.file_uploader(
        "内部文書をアップロード",
        type=["pdf", "txt", "docx"],
        key="internal_document"
    )
    
    if legal_doc and internal_doc:
        st.session_state.documents = {
            'legal': _save_uploaded_file(legal_doc),
            'internal': _save_uploaded_file(internal_doc)
        }
        st.success("文書が正常にアップロードされました。")
        st.session_state.current_step = 'analysis'

def _save_uploaded_file(uploaded_file) -> Optional[Path]:
    """アップロードされたファイルを一時ディレクトリに保存"""
    if uploaded_file is not None:
        try:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                return Path(tmp_file.name)
        except Exception as e:
            st.error(f"ファイルの保存中にエラーが発生しました: {str(e)}")
            return None
    return None
