# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# # Load FinBERT (or use 'bert-base-uncased' for general text)
# MODEL_NAME = "ProsusAI/finbert"

# # Load tokenizer and model
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)  # 3 labels: Positive, Negative, Neutral

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# model.to(device)


# def preprocess_text(text_list):
#     return tokenizer(text_list, padding=True, truncation=True, max_length=256, return_tensors="pt")

# # Example input
# texts = ["The stock market is performing well today!", 
#          "There are concerns about inflation causing volatility.", 
#          "No major movements in the stock market."]

# inputs = preprocess_text(texts)
# inputs = {key: value.to(device) for key, value in inputs.items()}

# with torch.no_grad():
#     outputs = model(**inputs)
#     predictions = torch.argmax(outputs.logits, dim=1)  # Get the highest probability class

# # Map numeric predictions to sentiment labels
# label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
# sentiments = [label_map[pred.item()] for pred in predictions]

# # Print results
# for text, sentiment in zip(texts, sentiments):
#     print(f"Text: {text}\nSentiment: {sentiment}\n")

import requests
import re
import torch
import numpy as np
import spacy
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load NLP models
nlp = spacy.load("en_core_web_sm")

# Load FinBERT for sentiment analysis
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

HEADERS = {"User-Agent": "YourName your@email.com"}

### Step 1: Extract Text from SEC Filing ###
def extract_text_from_sec_filing(filing_url):
    response = requests.get(filing_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main text from filing
    text = " ".join([p.get_text() for p in soup.find_all("p")])
    text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
    return text

### Step 2: Sentiment Analysis Using FinBERT ###
def get_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_labels[torch.argmax(predictions).item()]
    confidence = predictions.max().item()
    return sentiment, confidence

### Step 3: Extract Key Financial Terms using TF-IDF ###
def extract_keywords(text, n_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    top_n_indices = np.argsort(tfidf_matrix.toarray()[0])[-n_keywords:]
    keywords = feature_names[top_n_indices]
    return keywords.tolist()

### Step 4: Topic Modeling using LDA ###
def get_topics(text, n_topics=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([text])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(tfidf_matrix)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic in lda.components_:
        top_words = feature_names[np.argsort(topic)[-5:]]
        topics.append(", ".join(top_words))

    return topics

### Step 5: Named Entity Recognition (NER) ###
def extract_named_entities(text):
    doc = nlp(text)
    entities = {ent.label_: [] for ent in doc.ents}
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    
    return {key: list(set(value)) for key, value in entities.items()}  # Remove duplicates

### Step 6: Analyze SEC Filing ###
def analyze_sec_filing(filing_url):
    print(f"Analyzing filing: {filing_url}")

    # Extract text
    text = extract_text_from_sec_filing(filing_url)

    # Get sentiment
    sentiment, confidence = get_sentiment(text[:512])  # First 512 tokens for efficiency

    # Get keywords
    keywords = extract_keywords(text)

    # Get topics
    topics = get_topics(text)

    # Get named entities
    entities = extract_named_entities(text)

    return {
        "Sentiment": sentiment,
        "Sentiment Confidence": confidence,
        "Keywords": keywords,
        "Topics": topics,
        "Entities": entities,
    }

# Example Usage:
filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000010/a10-q20231230.htm"
result = analyze_sec_filing(filing_url)

# Print analysis
for key, value in result.items():
    print(f"{key}: {value}")
