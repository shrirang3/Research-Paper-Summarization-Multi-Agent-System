import streamlit as st
from PyPDF2 import PdfReader
import os
import tempfile
from agents.pdf_parser_summarizer import (
    process_documents,
    initialize_deepseek_llm,
    create_rag_chain
)

from audio_utils import (
    generate_podcast_script,
    text_to_speech,
    autoplay_audio,
    cleanup_audio_files
)

from agents.search_rp import search_arxiv

from agents.summary_generator import ArxivSummarizerAgent
from agents.classifier import ResearchPaperClassifier

import tempfile
import os


# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Initialize session state
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'current_interface' not in st.session_state:
    st.session_state.current_interface = "PDF"

# Sidebar with interface selection
with st.sidebar:
    st.header("Interface Selection")
    interface = st.radio(
        "Choose Interface",
        ["PDF Chat", "Text Analysis"],
        index=0 if st.session_state.current_interface == "PDF" else 1
    )

# Main content area
st.title("Research Paper Analyzer")

# PDF Chat Interface
if interface == "PDF Chat":
    st.session_state.current_interface = "PDF"
    
    # PDF Upload Section
    uploaded_file = st.file_uploader("Upload Research Paper (PDF)", type="pdf")
    
    if uploaded_file is not None:
        if not st.session_state.vector_store:
            with st.spinner("Processing PDF..."):
                try:
                    # Save uploaded file to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process PDF
                    st.session_state.vector_store = process_documents(tmp_path)
                    st.session_state.llm = initialize_deepseek_llm()
                    st.session_state.rag_chain = create_rag_chain(
                        st.session_state.vector_store,
                        st.session_state.llm
                    )
                    os.unlink(tmp_path)  # Cleanup temp file
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.session_state.vector_store = None

    if st.session_state.vector_store:
        # Chat Interface
        st.subheader("Paper Chat Assistant")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    with st.expander("Source Documents"):
                        for doc in message["sources"]:
                            st.write(f"**Page {doc.metadata.get('page', 'N/A')}**")
                            st.caption(doc.page_content[:300] + "...")
        
        # Chat input
        if prompt := st.chat_input("Ask about the paper..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.spinner("Generating response..."):
                try:
                    # Get RAG response
                    result = st.session_state.rag_chain.invoke({"query": prompt})
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["result"],
                        "sources": result["source_documents"]
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.session_state.current_interface = "Text"
    
    # Text Analysis Interface
    st.subheader("Text Analysis Interface")
    
    # Text input
    input_text = st.text_area(
        "Enter research paper text:",
        height=100,
        value=st.session_state.input_text
    )
    
    # Submit button
    if st.button("Submit Text"):
        st.session_state.input_text = input_text
        st.success("Text submitted!")
        
        try:
            # Search arXiv using the input text
            papers = search_arxiv(st.session_state.input_text)
            
            if papers:
                st.subheader("Search Results")
                
                # Display papers in pointwise format
                for idx, paper in enumerate(papers, 1):
                    st.markdown(f"{idx}. **{paper['title']}**")
                    st.caption(f"Published: {paper['published']}")
                
                # Store papers in session state for further processing
                st.session_state.papers = papers
            else:
                st.info("No papers found matching your query.")
                
        except Exception as e:
            st.error(f"Error searching arXiv: {str(e)}")
    
    if st.session_state.input_text:
        # Analysis controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Add number input before the button
            paper_number = st.number_input(
                "Paper number to summarize",
                min_value=1,
                max_value=4,
                value=1,
                key="summary_length"
            )
            st.session_state.paper_number = paper_number
            
            if st.button("Generate Summary"):
                summarizer = ArxivSummarizerAgent()
                #paper=papers[0]
                st.session_state.summary = summarizer.summarize_paper(st.session_state.papers[st.session_state.paper_number-1])
                # if st.session_state.input_text:
                #     st.session_state.summary = generate_summary(
                #         st.session_state.input_text,
                #         num_points=st.session_state.summary_length  # Pass to your function
                #     )
                # else:
                #     st.warning("Please submit text first!")
                # st.subheader("📄 Paper Summary")
                # st.write(st.session_state.summary)
    

        # Rest of your code remains unchanged
        
        with col2:
            
            categories_input = st.text_input("Enter categories (comma-separated):")
            categories_final = [cat.strip() for cat in categories_input.split(",") if cat.strip()]

            if st.button("Classify Text") and categories_input:
                classifier = ResearchPaperClassifier(categories_final, use_llm=True)
                result = classifier.classify(
                    title=st.session_state.papers[0]['title'],
                    abstract=st.session_state.papers[0]['abstract']
                )
                st.session_state.classification = result

        
        # Display results
        if 'summary' in st.session_state:
            st.subheader("Summary")
            st.write(st.session_state.summary)
        
        if 'classification' in st.session_state:
            st.subheader("Classification Result")
            st.write(f"Category: {st.session_state.classification['category']}")
            st.write(f"Confidence: {st.session_state.classification['confidence']:.2f}")

# Add this in your results section after summary and classification display
if 'summary' in st.session_state and 'classification' in st.session_state:
    st.divider()
    st.subheader("🎧 Audio Podcast")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("Generate Audio Summary"):
            try:
                # Generate podcast script
                podcast_script = generate_podcast_script(
                    st.session_state.summary,
                    st.session_state.classification
                )
                
                # Create audio file
                podcast_file = text_to_speech(podcast_script)
                st.session_state.podcast_generated = True
                st.session_state.podcast_file = podcast_file
                
            except Exception as e:
                st.error(f"Podcast generation failed: {str(e)}")
    
    if st.session_state.get('podcast_generated', False):
        with col2:
            st.markdown("### Podcast Controls")
            
            # Display audio player
            st.audio(st.session_state.podcast_file, format="audio/mp3")
            
            # Auto-play toggle
            auto_play = st.checkbox("Auto-play podcast")
            if auto_play:
                audio_html = autoplay_audio(st.session_state.podcast_file)
                st.components.v1.html(audio_html, height=0)
            
            # Download button
            with open(st.session_state.podcast_file, "rb") as f:
                st.download_button(
                    label="Download Podcast",
                    data=f,
                    file_name="research_podcast.mp3",
                    mime="audio/mp3"
                )
            
            # Cleanup when session ends
            cleanup_audio_files(st.session_state.podcast_file)