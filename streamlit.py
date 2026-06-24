import streamlit as st
import requests
import pandas as pd

st.title("RAG Application")

if "all_doc_ids" not in st.session_state:
    st.session_state.all_doc_ids = []
if "all_extracted_texts" not in st.session_state:
    st.session_state.all_extracted_texts = {}
if "all_extracted_info" not in st.session_state:
    st.session_state.all_extracted_info = {}

uploaded_files = st.file_uploader("upload_documents", accept_multiple_files=True)

if st.button("Upload!"):
    if uploaded_files:
        files = [("files", file) for file in uploaded_files]
        response = requests.post("http://localhost:8000/upload", files=files)
        
        data = response.json()
        st.session_state.all_doc_ids = data["document_ids"]
        st.session_state.all_extracted_texts = data["extracted_texts"]
        st.session_state.all_extracted_info = data["extracted_info"]
        st.success(f"✅ Uploaded {len(st.session_state.all_doc_ids)} documents!")
    else:
        st.warning("Select files first!")

if st.session_state.all_extracted_info:
    st.header("📊 Extracted Information")
    
    for doc_id in st.session_state.all_doc_ids:
        if doc_id in st.session_state.all_extracted_info:
            info = st.session_state.all_extracted_info[doc_id]
            
            with st.expander(f"📄 {info.get('type', 'Document')} - {doc_id[:8]}..."):
                df = pd.DataFrame(list(info.items()), columns=["Field", "Value"])
                st.table(df)

if st.session_state.all_extracted_texts:
    st.header("📄 View Extracted Text")
    
    view_option = st.radio(
        "Choose view option:",
        ["Single Document", "All Documents"]
    )
    
    if view_option == "Single Document":
        selected_doc = st.selectbox(
            "Select document",
            st.session_state.all_doc_ids
        )
        doc_id = selected_doc
        if st.session_state.all_extracted_texts.get(doc_id):
            st.text_area(
                f"Content: {doc_id[:8]}...",
                value=st.session_state.all_extracted_texts[doc_id],
                height=300,
                disabled=True
            )
    
    else:
        for doc_id in st.session_state.all_doc_ids:
            with st.expander(f"Document: {doc_id[:8]}..."):
                st.text_area(
                    f"Content",
                    value=st.session_state.all_extracted_texts[doc_id],
                    height=250,
                    disabled=True,
                    key=doc_id
                )

st.divider()

st.header("🔍 Search All Documents")

query = st.text_input("Enter query to search")

if st.button("Search"):
    if st.session_state.all_doc_ids:
        if query:
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "query": query,
                    "top_k": 10
                }
            )
            result = response.json()
            st.markdown("### Answer:")
            st.write(result.get("answer", result))
        else:
            st.warning("Enter a query first!")
    else:
        st.warning("Upload documents first!")