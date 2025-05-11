from agents.search_rp import search_arxiv
from agents.summary_generator import ArxivSummarizerAgent
from agents.classifier import ResearchPaperClassifier
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

query = "Deep learning"  # Attention Is All You Need paper
papers = search_arxiv(query)

print(papers[0]['title']) #We get title of paper

if papers:
    summarizer=ArxivSummarizerAgent()
    
    #print(summarizer.summarize_paper(papers[0])) # We get summary of paper
else:
    print("No papers found.")


# Define your categories
PAPER_CATEGORIES = [
    "NLP", 
    "Computer Vision",
    "Machine Learning",
    "Neuroscience",
    "Robotics"
]

# Initialize classifier



classifier = ResearchPaperClassifier(PAPER_CATEGORIES, use_llm=True)
# Get classification
result = classifier.classify(
    title=papers[0]['title'],
    abstract=papers[0]['abstract']
)

print("Classification Result:")
print(result)

