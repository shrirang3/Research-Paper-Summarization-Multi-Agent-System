
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import re

# Configure API wrapper
api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=4,
    doc_content_chars_max=4000,
    sort_by="relevance",
    load_max_docs=4
)

arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

def search_arxiv(query: str) -> list[dict]:
    """Search arXiv and return list of papers with metadata"""
    results = arxiv_tool.invoke(query)
    
    # Regex pattern to extract paper components
    paper_pattern = r'Published: (.*?)\nTitle: (.*?)\nAuthors: (.*?)\nSummary: (.*?)(?=\nPublished:|\Z)'
    
    papers = re.findall(paper_pattern, results, re.DOTALL)
    
    return [{
        'title': paper[1].strip(),
        'abstract': paper[3].strip(),
        'published': paper[0].strip(),
        'authors': paper[2].strip().split(', ')
    } for paper in papers]

# Example usage
if __name__ == "__main__":
    query = "Deep Learning"  # Attention Is All You Need paper
    papers = search_arxiv(query)
    
    if papers:
        print(f"Found {len(papers)} papers:\n")
        for idx, paper in enumerate(papers, 1):
            print(f"{idx}. {paper['title']}")
            print(f"Abstract: {paper['abstract'][:150]}...\n")  # Show first 150 chars
    else:
        print("No papers found.")