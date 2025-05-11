from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def process_documents(pdf_path: str) -> FAISS:
    """Process PDF and create FAISS vector store"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

def initialize_deepseek_llm():
    """Initialize DeepSeek model through Ollama"""
    return Ollama(
        model="deepseek-r1:latest",  # Ensure this model is available in Ollama
        temperature=0.3,
        system="You are a technical research assistant. Provide detailed answers based on the context."
    )

def create_rag_chain(vector_store: FAISS, llm):
    """Create RAG pipeline with custom prompt"""
    template = """[INST] <<SYS>>
    Use the following context to answer the question. Cite sources using [page number].
    Be technical and precise. If unsure, state that.
    <</SYS>>

    Context: {context}

    Question: {question} [/INST]"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# Example usage
if __name__ == "__main__":
    # 1. Process documents
    vector_store = process_documents("C:/Users/shrir/Downloads/U-Net_and_Its_Variants_for_Medical_Image_Segmentation_A_Review_of_Theory_and_Applications.pdf")
    
    # 2. Initialize LLM
    llm = initialize_deepseek_llm()
    
    # 3. Create RAG chain
    qa_chain = create_rag_chain(vector_store, llm)
    
    # 4. Query the system
    result = qa_chain.invoke({
        "query": "Summarize this paper"
    })
    
    print("Answer:", result["result"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(f"- Page {doc.metadata['page']}: {doc.page_content[:100]}...")