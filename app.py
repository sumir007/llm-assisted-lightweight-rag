import streamlit as st
import os
import tempfile
import csv
import json
from datetime import datetime
from io import StringIO
from rag_cli import (
    load_chunks, find_top_chunks, build_user_prompt, trim_to_fit, 
    call_github_models, SYSTEM_PROMPT, TOP_K, MAX_INPUT_CHARS
)

st.set_page_config(page_title="Contract RAG Analyzer", layout="wide")
st.title("📄 Contract RAG Analyzer")
st.write("Upload contracts and ask questions about dates, terms, validity, and more.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = os.getenv("GITHUB_PAT", "")
    if api_key:
        st.success("Online")
    else:
        st.warning("⚠️ API key not detected.")
    
    st.markdown("---")
    st.markdown("**Suggested Queries:**")
    st.markdown("- 'What is the start date?'")
    st.markdown("- 'What is the contract term?'")
    st.markdown("- 'Is it valid or expired?'")

# Initialize session state for chunks and results
if 'cached_chunks' not in st.session_state:
    st.session_state.cached_chunks = None
    st.session_state.cached_file_names = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Main content
st.header("📤 Upload Documents")
uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type="pdf")

chunks = []
if uploaded_files:
    # Check if files changed
    current_file_names = [f.name for f in uploaded_files]
    if current_file_names != st.session_state.cached_file_names:
        # Files changed, reload chunks
        st.session_state.batch_results = []  # Reset results for new files
        temp_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_paths.append(temp_file.name)
        st.session_state.cached_chunks = load_chunks(temp_paths)
        st.session_state.cached_file_names = current_file_names
        st.success(f"✓ Loaded {len(st.session_state.cached_chunks)} chunks from {len(uploaded_files)} PDF(s)")
    else:
        st.info(f"✓ Using cached chunks ({len(st.session_state.cached_chunks)} total)")
    
    chunks = st.session_state.cached_chunks
    
    with st.expander("📊 Chunk Details"):
        st.info(f"Total chunks created: {len(chunks)}\nChunks per file: ~{len(chunks)//len(uploaded_files) if uploaded_files else 0}")
else:
    st.info("👆 Please upload at least one PDF to begin")

st.markdown("---")
st.header("❓ Ask a Question")
query = st.text_input(
    "Enter your query",
    placeholder="e.g., What is the agreement date and contract term?"
)

col1, col2, col3 = st.columns(3)
with col1:
    start_button = st.button("🚀 Start Processing", use_container_width=True)

if start_button:
    if not query:
        st.error("❌ Please enter a query.")
    elif not chunks:
        st.error("❌ No documents loaded. Please upload PDFs first.")
    else:
        with st.spinner("⏳ Processing your query..."):
            try:
                # Call your RAG logic
                st.info(f"ℹ️ Total chunks loaded: {len(chunks)}")
                top_chunks = find_top_chunks(chunks, query, TOP_K)
                st.info(f"✓ Retrieved {len(top_chunks)} chunks (TOP_K={TOP_K})")
                
                if not top_chunks:
                    st.warning("⚠️ No relevant chunks found. Try a different query or check your documents.")
                else:
                    # Show retrieved chunks (debug info)
                    with st.expander(f"🔍 Retrieved Chunks ({len(top_chunks)} found)"):
                        for i, chunk in enumerate(top_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk[:300])
                            st.divider()
                    
                    user_prompt = build_user_prompt(query, top_chunks)
                    user_prompt = trim_to_fit(SYSTEM_PROMPT, user_prompt, MAX_INPUT_CHARS)
                    
                    answer = call_github_models(SYSTEM_PROMPT, user_prompt)
                    
                    # Display result
                    st.markdown("---")
                    st.header("📋 Result")
                    st.text_area("Answer", value=answer, height=200, disabled=True, key="result")
                    
                    # Save to batch results
                    file_names = ", ".join([f.name for f in uploaded_files]) if uploaded_files else "Unknown"
                    st.session_state.batch_results.append({
                        "timestamp": datetime.now().isoformat(),
                        "file": file_names,
                        "query": query,
                        "answer": answer
                    })
                    
                    st.success(f"✓ Result #{len(st.session_state.batch_results)} saved to batch")
                    
                    # Copy button
                    st.caption("💡 Tip: Use Ctrl+A to select and copy the result")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Check your GitHub API key and try again.")

# Batch processing and export section
st.markdown("---")
st.header("📊 Batch Results")

if st.session_state.batch_results:
    st.info(f"✓ {len(st.session_state.batch_results)} result(s) in batch")
    
    # Display batch results
    with st.expander("View Batch Results"):
        for idx, result in enumerate(st.session_state.batch_results, 1):
            st.markdown(f"**Result {idx}** - {result['timestamp']}")
            st.markdown(f"📄 File: {result['file']}")
            st.markdown(f"❓ Query: {result['query']}")
            st.markdown(f"💬 Answer: {result['answer']}")
            st.divider()
    
    # Export to CSV
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Generate CSV Export"):
            # Create CSV content
            csv_filename = f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_lines = [["Timestamp", "File", "Query", "Answer"]]
            for result in st.session_state.batch_results:
                csv_lines.append([
                    result["timestamp"],
                    result["file"],
                    result["query"],
                    result["answer"]
                ])
            
            # Convert to CSV string
            output = StringIO()
            writer = csv.writer(output)
            writer.writerows(csv_lines)
            csv_string = output.getvalue()
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_string,
                file_name=csv_filename,
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Clear Batch"):
            st.session_state.batch_results = []
            st.rerun()
else:
    st.info("No results in batch yet. Process queries to populate batch results.")