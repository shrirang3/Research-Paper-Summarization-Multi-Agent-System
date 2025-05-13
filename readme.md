
# Research Paper Summarization Multi Agent System 📝🚀

This multi-agent system is an intelligent multi-agent system designed to automate the end-to-end summarization of research papers into engaging audio podcast summaries. It allows users to input academic papers via direct PDF upload, URL, or keyword-based ArXiv search. The system leverages state-of-the-art NLP models for document understanding, semantic chunking, and summarization. It then uses text-to-speech (TTS) technology to convert the textual summary into a clear, human-like audio narration. 

🖼️ **Results and UI screenshots** can be found in output folder.


## 🔑 Features

- 📄 **Multi-Input Support**: Accepts research papers via PDF upload, ArXiv keyword search, or direct URL.
- 🧠 **LLM-Based Summarization**: Uses powerful language models (e.g., Groq, Ollama) to generate concise and coherent summaries.
- 🧩 **Semantic Chunking**: Splits content intelligently using FAISS and HuggingFace Embeddings for better context preservation.
- 🔊 **Audio Podcast Generation**: Converts summaries into human-like speech using `gTTS` for easy listening.
- 🔍 **ArXiv Integration**: Supports direct search and retrieval from ArXiv using the Arxiv API.
- ⚙️ **Modular Multi-Agent Design**: Built with extensible agents, allowing integration of more capabilities like Q&A, critique, and translation.

## 🛠️ Technologies Used

- 📄 **RAG Pipeline with FAISS & DeepSeek LLM**: For PDF inputs, a Retrieval-Augmented Generation pipeline is used with FAISS vector store and DeepSeek LLM for summarization and classification.
- 🔍 **ArXiv Integration via search_rp Agent**: Fetches top 4 relevant or recent papers using the ArxivAPI wrapper; user selects one for further processing.
- 🤖 **Summarization & Classification Agents**: Summarize and classify selected papers using Groq LLM based on paper title and abstract.
- 🔊 **Audio Generation with gTTS**: Converts summarized text into natural-sounding audio podcasts.
- 🧠 **Modular Multi-Agent Architecture**: Cleanly separates concerns like search, summarization, classification, and audio generation for scalability.




## 🧪 Setting Up the Virtual Environment (Using Conda)

1. **Create a new Conda environment**  

   ```bash
   conda create -n venv python=3.10
   conda activate venv
   ```
  
2. **Clone repo and Install dependancies** 
    ```bash
    git clone https://github.com/shrirang3/Research-Paper-Summarization-Multi-Agent-System.git
    pip install -r requirements.txt
    ```

## ⚙️ LLM Configuration Guide

### A. 🧠 Ollama & DeepSeek Setup

1. **Install Ollama**  
   Download and install from the official site:  
   👉 [https://ollama.com/download](https://ollama.com/download)

2. **Pull the DeepSeek model**  
   ```bash
   ollama pull deepseek-coder:latest
    ```
### B. 🔐 Groq API Key Generation

Follow these steps to generate and use your Groq API key:

1. **Sign Up or Log In**
   - Visit the Groq Console:  
     👉 [https://console.groq.com](https://console.groq.com)

2. **Generate a New API Key**
   - Go to the **API Keys** section in the dashboard.
   - Click on **"Create API Key"**.


3. **Add Key in environment variables**
    

## 🔗 References

  - [Groq API Documentation](https://console.groq.com/docs)

  - [Streamlit Docs](https://docs.streamlit.io/)

  - [RAG Pipeline](https://sebastian-petrus.medium.com/developing-rag-systems-with-deepseek-r1-ollama-f2f561cfda97)
  - [Gtts Usage](https://gtts.readthedocs.io/en/latest/)
