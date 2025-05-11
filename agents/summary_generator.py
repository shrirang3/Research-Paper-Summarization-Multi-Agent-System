from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os
from search_rp import search_arxiv  # Assuming this function is defined in search_rp.py

from dotenv import load_dotenv
import os
load_dotenv()

class ArxivSummarizerAgent:
    def __init__(self):
        self.summary_template = """You are a research paper summarization expert. 
        Create a structured summary of this paper with the following sections:
        1. Core Contribution (1 sentence)

        Paper Title: {title}
        Abstract: {abstract}"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical research assistant skilled in summarizing academic papers."),
            ("human", self.summary_template)
        ])
        
        # Initialize ChatGroq with Qwen-32B model
        self.llm = ChatGroq(
            model_name="qwen-qwq-32b",  # Verify exact model name on Groq
            temperature=0.2,
            max_tokens=1024
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def summarize_paper(self, paper_data: dict) -> str:
        """Summarize a paper using Groq/Qwen"""
        try:
            return self.chain.invoke({
                "title": paper_data['title'],
                "abstract": paper_data['abstract']
            })['text']
        except Exception as e:
            return f"Error generating summary: {str(e)}"

# Usage (ensure GROQ_API_KEY is set in environment)
# Corrected environment variable access
if __name__ == "__main__":
    # Get API key from .env file
    groq_api_key = os.getenv("GROQ_API_KEY")  # Use parentheses () not square brackets []
    
    # Set up Groq
    os.environ["GROQ_API_KEY"] = groq_api_key
    
    papers = search_arxiv("1706.03762")
    
    if papers:
        summarizer = ArxivSummarizerAgent()
        first_paper = papers[0]
        
        print("Paper Title:", first_paper['title'])
        print("\nSummary:")
        print(summarizer.summarize_paper(first_paper))