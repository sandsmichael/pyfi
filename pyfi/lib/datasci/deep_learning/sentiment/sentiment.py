import requests
import json
import pandas as pd
import torch
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# Load FinBERT for financial sentiment analysis
MODEL_NAME = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# VADER Sentiment Analyzer for social media
vader = SentimentIntensityAnalyzer()

HEADERS = {"User-Agent": "YourName your@email.com"}

### 1️⃣ Fetch Market News Headlines ###
def get_market_news():
    url = "https://finance.yahoo.com/news"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print("Failed to fetch data.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    
    headlines = []
    
    # Yahoo Finance news headlines are inside <h3> tags with specific class names
    for h3 in soup.find_all("h3", class_="Mb(5px)"):
        a_tag = h3.find("a")
        if a_tag:
            headlines.append(a_tag.get_text())

    return headlines[:10]  # Get top 10 headlines

### 2️⃣ Analyze Sentiment with FinBERT ###
def get_finbert_sentiment(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_labels[torch.argmax(predictions).item()]
    confidence = predictions.max().item()
    return sentiment, confidence

### 3️⃣ Analyze Sentiment of Tweets/Reddit Posts with VADER ###
def get_vader_sentiment(text):
    score = vader.polarity_scores(text)
    return "Positive" if score["compound"] > 0.05 else "Negative" if score["compound"] < -0.05 else "Neutral"

### 4️⃣ Get Overall Market Sentiment ###
def get_market_sentiment():
    print("📢 Fetching market news...")
    news_headlines = get_market_news()
    print(news_headlines)

    print("📝 Analyzing sentiment...")
    news_sentiments = [get_finbert_sentiment(headline)[0] for headline in news_headlines]

    print("📊 Aggregating sentiment...")
    sentiment_counts = pd.Series(news_sentiments).value_counts(normalize=True) * 100

    return {
        "Positive": sentiment_counts.get("Positive", 0),
        "Neutral": sentiment_counts.get("Neutral", 0),
        "Negative": sentiment_counts.get("Negative", 0)
    }

# Run the market sentiment tool
market_sentiment = get_market_sentiment()
print("📊 Market Sentiment Score:", market_sentiment)
