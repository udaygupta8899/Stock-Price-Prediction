from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class FinBERTAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of financial text using FinBERT
        Returns: sentiment (str), confidence (float)
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get the predicted class and confidence
        predicted_class = torch.argmax(predictions).item()
        confidence = predictions[0][predicted_class].item()
        
        # Map the class index to sentiment
        sentiment_map = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }
        
        return sentiment_map[predicted_class], confidence

    def analyze_batch(self, texts):
        """
        Analyze sentiment for a batch of texts
        Returns: list of (sentiment, confidence) tuples
        """
        results = []
        for text in texts:
            sentiment, confidence = self.analyze_sentiment(text)
            results.append((sentiment, confidence))
        return results 