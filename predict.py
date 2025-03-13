import torch
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from tft_model import TemporalFusionTransformer
import joblib
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, model_path, scaler_path):
        # Load model
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Initialize BERT for sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Initialize News API client
        self.newsapi = NewsApiClient(api_key='813bb17cd2704c12a2acf66732f973bc')  # Replace with your API key
        
    def get_stock_data(self, symbol, period='1y'):
        """Get stock data from yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            # Select relevant columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            return df
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def get_news_data(self, symbol, days=30):
        """Get news articles using News API"""
        try:
            # Get news articles
            news = self.newsapi.get_everything(
                q=symbol,
                from_param=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                to=datetime.now().strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt'
            )
            
            # Extract titles and descriptions
            articles = []
            for article in news['articles']:
                text = f"{article['title']} {article['description']}"
                articles.append(text)
            
            return articles
        except Exception as e:
            print(f"Error fetching news data: {e}")
            return []
    
    def get_sentiment_embedding(self, text):
        """Get sentiment embedding for a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def prepare_data(self, df, sentiment_data, sequence_length=10):
        """Prepare data for prediction"""
        # Scale the data
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Get the last sequence
        sequence = df.iloc[-sequence_length:]
        sentiment_sequence = sentiment_data[-sequence_length:]
        
        # Prepare features
        features = {
            'open': torch.FloatTensor(sequence['Open'].values).unsqueeze(0),
            'high': torch.FloatTensor(sequence['High'].values).unsqueeze(0),
            'low': torch.FloatTensor(sequence['Low'].values).unsqueeze(0),
            'close': torch.FloatTensor(sequence['Close'].values).unsqueeze(0),
            'volume': torch.FloatTensor(sequence['Volume'].values).unsqueeze(0),
        }
        
        # Prepare sentiment
        sentiment = torch.FloatTensor(sentiment_sequence).unsqueeze(0)
        
        return features, sentiment
    
    def predict(self, symbol):
        """Make prediction for a specific stock"""
        # Get stock data
        df = self.get_stock_data(symbol)
        if df is None:
            return None
        
        # Get news data
        news_articles = self.get_news_data(symbol)
        
        # Get sentiment embeddings for all articles
        sentiment_embeddings = []
        for article in news_articles:
            embedding = self.get_sentiment_embedding(article)
            sentiment_embeddings.append(embedding)
        
        # If no news articles, use zero embedding
        if not sentiment_embeddings:
            sentiment_embeddings = [np.zeros(768)]
        
        # Prepare data
        features, sentiment = self.prepare_data(df, sentiment_embeddings)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(features, sentiment)
        
        # Inverse transform the prediction
        prediction = self.scaler.inverse_transform([[prediction.item()]])[0][0]
        
        return {
            'predicted_price': prediction,
            'current_price': df['Close'].iloc[-1],
            'news_count': len(news_articles)
        }

def main():
    # Example usage
    predictor = StockPredictor(
        model_path="tft_model.ckpt",
        scaler_path="scaler.joblib"
    )
    
    # Example prediction
    symbol = "RELIANCE.NS"  # NSE symbol
    result = predictor.predict(symbol)
    
    if result:
        print(f"Predicted price for {symbol}: {result['predicted_price']:.2f}")
        print(f"Current price: {result['current_price']:.2f}")
        print(f"Number of news articles analyzed: {result['news_count']}")
    else:
        print("Failed to make prediction")

if __name__ == "__main__":
    main()
