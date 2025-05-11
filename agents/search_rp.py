from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
import re

# Configure API wrapper with higher character limit and proper sorting
api_wrapper_arxiv = ArxivAPIWrapper(
    top_k_results=4,  # Number of papers to return
    doc_content_chars_max=4000,  # Increased to handle multiple papers
    sort_by="relevance",
    load_max_docs=4  # Ensure we load all requested papers
)

arxiv_tool = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

def search_arxiv(query: str) -> str:
    """Search arXiv and return formatted results"""
    return arxiv_tool.invoke(query)

# Get papers with better query formatting
results = search_arxiv("1706.03762")

# Improved regex pattern to extract titles
titles = re.findall(r'Title: (.*?)\n', results)

# Print results with numbering
print(f"Found {len(titles)} papers:\n")
for i, title in enumerate(titles, 1):
    print(f"{i}. {title.strip()}")