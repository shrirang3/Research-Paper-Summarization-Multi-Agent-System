# from langchain_community.tools import ArxivQueryRun
# from langchain_community.utilities import ArxivAPIWrapper
# import re

# # Configure API wrapper with higher character limit and proper sorting
# api_wrapper_arxiv = ArxivAPIWrapper(
#     top_k_results=4,  # Number of papers to return
#     doc_content_chars_max=4000,  # Increased to handle multiple papers
#     sort_by="relevance",
#     load_max_docs=4  # Ensure we load all requested papers
# )

# arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# def search_arxiv(query: str) -> str:
#     """Search arXiv and return formatted results"""
#     return arxiv_tool.invoke(query)

# # Get papers with better query formatting
# results = search_arxiv("1706.03762")

# # Improved regex pattern to extract titles
# titles = re.findall(r'Title: (.*?)\n', results)

# # Print results with numbering
# print(f"Found {len(titles)} papers:\n")
# for i, title in enumerate(titles, 1):
#     print(f"{i}. {title.strip()}")

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
    query = "1706.03762"  # Attention Is All You Need paper
    papers = search_arxiv(query)
    
    if papers:
        print(f"Found {len(papers)} papers:\n")
        for idx, paper in enumerate(papers, 1):
            print(f"{idx}. {paper['title']}")
            print(f"Abstract: {paper['abstract'][:150]}...\n")  # Show first 150 chars
    else:
        print("No papers found.")