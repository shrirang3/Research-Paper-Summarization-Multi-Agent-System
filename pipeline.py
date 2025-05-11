from agents.search_rp import search_arxiv
from agents.summary_generator import ArxivSummarizerAgent
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

query = "1706.03762"  # Attention Is All You Need paper
papers = search_arxiv(query)

#print(papers[0]['title']) We get title of paper

if papers:
    summarizer=ArxivSummarizerAgent()
    first_paper = papers[0]
    print(summarizer.summarize_paper(first_paper)) # We get summary of paper
else:
    print("No papers found.")
