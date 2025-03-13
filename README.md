# Stock Price Analysis and Prediction System

A comprehensive stock analysis system that combines real-time market data with sentiment analysis for Indian stocks.

## Features

- Historical data analysis with interactive charts
- Technical indicators visualization:
  - RSI (Relative Strength Index)
  - Moving Averages
  - Bollinger Bands
  - MACD (Moving Average Convergence Divergence)
- News sentiment analysis using FinBERT
- Real-time stock data updates
- Interactive dashboard with multiple views

## Setup

1. Clone the repository:
```bash
git clone https://github.com/udaygupta8899/stock-price-prediction.git
cd stock-price-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory and add your API keys:
```
NEWS_API_KEY=your_news_api_key
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Select a stock from the sidebar to view:
   - Historical price data and technical indicators
   - News sentiment analysis
   - Market trends and patterns

## Project Structure

- `app.py`: Main Streamlit application
- `finbert_analyzer.py`: Sentiment analysis using FinBERT
- `stock_list.json`: List of Indian stocks to analyze
- `dashboard_integration.py`: Dashboard components and visualizations
- `requirements.txt`: Python dependencies

## Technical Analysis

The system provides various technical indicators to help analyze stock trends:

1. **RSI (Relative Strength Index)**: Measures momentum and identifies overbought/oversold conditions
2. **Moving Averages**: Shows trend direction and potential support/resistance levels
3. **Bollinger Bands**: Indicates volatility and potential price breakouts
4. **MACD**: Identifies momentum changes and potential trend reversals

## Sentiment Analysis

The system uses FinBERT, a financial domain-specific BERT model, to analyze news sentiment:

- Processes financial news and social media data
- Classifies sentiment as positive, negative, or neutral
- Provides sentiment scores for better decision making

## Contributing

Feel free to submit issues and enhancement requests! 
