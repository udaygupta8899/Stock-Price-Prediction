import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
from time import sleep
from random import randint
from finbert_analyzer import FinBERTAnalyzer
import torch
from predict import StockPredictor
import numpy as np

# Set page config
st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .divider {
        height: 2px;
        background-color: #4CAF50;
        margin: 20px 0;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize FinBERT analyzer and TFT predictor
@st.cache_resource
def get_finbert_analyzer():
    return FinBERTAnalyzer()

@st.cache_resource(hash_funcs={
    "torch._C._TensorBase": lambda _: None,
    "torch.nn.parameter.Parameter": lambda _: None,
    "_torch.torch.classes": lambda _: None,
    "torch._classes": lambda _: None
})
def get_stock_predictor():
    return StockPredictor(
        model_path="tft_model.ckpt",
        scaler_path="scaler.joblib"
    )

def get_stock_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def get_news(symbol):
    """Fetch news articles for the selected stock"""
    try:
        # Replace with your News API key
        api_key = "813bb17cd2704c12a2acf66732f973bc"
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=5"
        response = requests.get(url)
        news_data = response.json()
        
        if news_data.get('status') == 'ok' and news_data.get('articles'):
            return news_data['articles']
        return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

def main():
    st.title("📈 Stock Market Dashboard")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Stock selection
    stock_list = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS"
    ]
    selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
    
    # Time period selection
    time_period = st.sidebar.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=3
    )
    
    # Fetch stock data
    df = get_stock_data(selected_stock, time_period)
    
    if df is not None and not df.empty:
        # Display stock information
        st.header(f"{selected_stock.replace('.NS', '')} Stock Analysis")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['Close'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Price</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{current_price:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            daily_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
            color = "green" if daily_change >= 0 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Daily Change</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {color}">
                    {daily_change:+.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume = df['Volume'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Volume</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{volume:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            high = df['High'].iloc[-1]
            low = df['Low'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3>High/Low</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{high:,.2f}/₹{low:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Price chart
        st.subheader("Price Chart")
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add Prediction Section
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Price Prediction")
        
        with st.spinner("Generating price prediction..."):
            try:
                predictor = get_stock_predictor()
                prediction_result = predictor.predict(selected_stock)
                
                if prediction_result:
                    current_price = prediction_result['current_price']
                    predicted_price = prediction_result['predicted_price']
                    news_count = prediction_result['news_count']
                    
                    # Calculate percentage change
                    price_change = ((predicted_price - current_price) / current_price) * 100
                    
                    # Create prediction cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Current Price</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{current_price:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Price</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{predicted_price:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        color = "green" if price_change >= 0 else "red"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Change</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {color}">
                                {price_change:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add prediction confidence indicator with news status
                    confidence_text = "Based on technical analysis only" if news_count == 0 else f"Based on analysis of {news_count} recent news articles"
                    confidence_color = "orange" if news_count == 0 else "green"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediction Details</h3>
                        <p style="color: {confidence_color}">{confidence_text}</p>
                        <p>Model confidence: {abs(price_change):.1f}%</p>
                        <p style="font-size: 0.9rem; color: #888;">
                            {f"Note: No recent news articles found. Prediction based on historical price patterns." if news_count == 0 else ""}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add prediction chart
                    fig_pred = go.Figure()
                    
                    # Add historical price data
                    fig_pred.add_trace(go.Scatter(
                        x=df.index[-30:],  # Last 30 days
                        y=df['Close'].iloc[-30:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='gray', width=1)
                    ))
                    
                    # Add current price point
                    fig_pred.add_trace(go.Scatter(
                        x=[df.index[-1]],
                        y=[current_price],
                        mode='markers',
                        name='Current Price',
                        marker=dict(size=10, color='blue')
                    ))
                    
                    # Add predicted price point
                    next_date = df.index[-1] + pd.Timedelta(days=1)
                    fig_pred.add_trace(go.Scatter(
                        x=[next_date],
                        y=[predicted_price],
                        mode='markers',
                        name='Predicted Price',
                        marker=dict(size=10, color='green' if price_change >= 0 else 'red')
                    ))
                    
                    # Add line connecting points
                    fig_pred.add_trace(go.Scatter(
                        x=[df.index[-1], next_date],
                        y=[current_price, predicted_price],
                        mode='lines',
                        name='Price Movement',
                        line=dict(color='green' if price_change >= 0 else 'red', dash='dash')
                    ))
                    
                    fig_pred.update_layout(
                        template="plotly_dark",
                        height=400,
                        showlegend=True,
                        margin=dict(l=20, r=20, t=40, b=20),
                        title="Price Prediction Trend",
                        xaxis_title="Date",
                        yaxis_title="Price (₹)"
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                else:
                    st.warning("Unable to generate prediction at this time. Please try again later.")
                    
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.info("""
                This could be due to:
                - Network connectivity issues
                - Limited availability of market data
                - Model loading issues
                Please try again in a few minutes.
                """)
        
        # News Section
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Latest News")
        
        news_articles = get_news(selected_stock)
        if news_articles:
            analyzer = get_finbert_analyzer()
            
            for article in news_articles:
                with st.expander(f"📰 {article['title']}", expanded=False):
                    st.write(article['description'])
                    st.markdown(f"[Read more]({article['url']})")
                    
                    # Analyze sentiment
                    try:
                        sentiment = analyzer.analyze(article['title'] + " " + article['description'])
                        sentiment_color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
                        st.markdown(f"Sentiment: <span style='color: {sentiment_color}'>{'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}</span>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning("Unable to analyze sentiment for this article.")
        else:
            st.info("No recent news articles found for this stock.")
    
    else:
        st.error("Unable to fetch stock data. Please try again later.")

if __name__ == "__main__":
    main() 
