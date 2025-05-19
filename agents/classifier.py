from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

# Hybrid classifier combining LLM and traditional ML
class ResearchPaperClassifier:
    def __init__(self, categories, use_llm=True):
        self.categories = categories
        self.use_llm = use_llm
        
        if use_llm:
            # LLM-based classifier
            self.llm = ChatGroq(
                model_name="qwen-qwq-32b",
                temperature=0.1
            )
            self.prompt = ChatPromptTemplate.from_template(
                """Classify this research paper into one of {categories}:
                Title: {title}
                Abstract: {abstract}
                
                Use this format:
                Category: <category>
                Confidence: <confidence_score>
                Reasoning: <brief explanation>
                
                Examples:
                - Category: NLP
                  Confidence: 0.9
                  Reasoning: Focuses on transformer architectures for language tasks
                
                - Category: Computer Vision
                  Confidence: 0.85  
                  Reasoning: Proposes new image segmentation technique"""
            )
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            # Traditional ML classifier
            self.vectorizer = TfidfVectorizer()
            self.model = LogisticRegression(multi_class='multinomial')

    def classify(self, title: str, abstract: str, keywords: list = []):
        if self.use_llm:
            return self._llm_classify(title, abstract)
        else:
            return self._ml_classify(title, abstract, keywords)

    def _llm_classify(self, title: str, abstract: str):
        response = self.chain.invoke({
            "title": title,
            "abstract": abstract,
            "categories": ", ".join(self.categories)
        })
        return self._parse_llm_response(response)

    def _parse_llm_response(self, response: str):
        # Extract classification from LLM response
        category = None
        confidence = None
        reasoning = None
        
        for line in response.split('\n'):
            if line.startswith('Category:'):
                category = line.split(': ')[1].strip()
            elif line.startswith('Confidence:'):
                confidence = float(line.split(': ')[1])
            elif line.startswith('Reasoning:'):
                reasoning = line.split(': ')[1].strip()
                
        return {
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning
        }

    # # Traditional ML methods
    # def train_ml_model(self, X_train, y_train):
    #     """Train on labeled data (text, labels)"""
    #     text_data = [f"{title} {abstract}" for title, abstract in X_train]
    #     X_vec = self.vectorizer.fit_transform(text_data)
    #     self.model.fit(X_vec, y_train)

    # def _ml_classify(self, title: str, abstract: str, keywords: list):
    #     text = f"{title} {abstract} {' '.join(keywords)}"
    #     X_vec = self.vectorizer.transform([text])
    #     probas = self.model.predict_proba(X_vec)[0]
        
    #     return {
    #         "predictions": sorted([
    #             {"category": cat, "probability": float(prob)}
    #             for cat, prob in zip(self.categories, probas)
    #         ], key=lambda x: x['probability'], reverse=True)
    #     }

# Usage Example
if __name__ == "__main__":

    groq_api_key = os.getenv("GROQ_API_KEY")  # Use parentheses () not square brackets []
    
    # Set up Groq
    os.environ["GROQ_API_KEY"] = groq_api_key

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

    # Sample paper data (arXiv ID 1706.03762 - "Attention Is All You Need")
    paper = {
        "title": "Attention Is All You Need",
        "abstract": """We propose a new simple network architecture, the Transformer, 
        based solely on attention mechanisms, dispensing with recurrence and convolutions..."""
    }

    # Get classification
    result = classifier.classify(
        title=paper['title'],
        abstract=paper['abstract']
    )
    
    print("Classification Result:")
    print(f"Category: {result['category']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Reasoning: {result['reasoning']}")