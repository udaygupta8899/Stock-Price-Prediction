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
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize FinBERT analyzer
@st.cache_resource
def get_finbert_analyzer():
    return FinBERTAnalyzer()

# Initialize Stock Predictor with cache configuration to prevent PyTorch errors
@st.cache_resource(hash_funcs={
    "torch._C._TensorBase": lambda _: None,
    "torch.nn.parameter.Parameter": lambda _: None,
    "_torch.torch.classes": lambda _: None,
    "torch._classes": lambda _: None
}, ttl=1)  # Adding a short time-to-live to refresh predictor
def get_stock_predictor():
    # Version 2: Now supports external_news parameter
    print("Initializing new StockPredictor instance...")
    return StockPredictor(
        model_path="tft_model.ckpt",
        scaler_path="scaler.joblib"
    )

# Set page config for wide mode and title
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide",  # Enable wide mode by default
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    :root {
        --primary: #4FC3F7;
        --background: #0E1117;  /* Dark background */
        --card-bg: rgba(255, 255, 255, 0.05);  /* Transparent card */
        --text-color: #ffffff;
        --hover-color: #4FC3F7;
    }

    .stApp {
        background: var(--background);
        color: var(--text-color);
        font-family: 'Segoe UI', sans-serif;
    }

    .metric-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }

    .news-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .news-card:hover {
        transform: translateY(-3px);
    }

    h1, h2, h3 {
        color: var(--hover-color) !important;
        margin-bottom: 1rem !important;
    }

    a {
        color: var(--hover-color);
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, var(--hover-color) 0%, transparent 100%);
        margin: 2rem 0;
    }

    .st-bb { background-color: transparent; }
    .st-at { background-color: var(--hover-color) !important; }
</style>
""", unsafe_allow_html=True)

# Stock list
all_stocks = {
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Reliance Industries": "RELIANCE.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Tata Consultancy Services": "TCS.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Nestle India": "NESTLEIND.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Wipro": "WIPRO.NS",
    "Tech Mahindra": "TECHM.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Transmission": "ADANITRANS.NS",
    "GAIL": "GAIL.NS",
    "NTPC": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Cipla": "CIPLA.NS",
    "Grasim Industries": "GRASIM.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Titan Company": "TITAN.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Zee Entertainment": "ZEEL.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    # Add more stocks if needed...
}

# Sidebar Configuration
st.sidebar.title("📈 Stock Dashboard")
st.sidebar.markdown("---")

# Add a button to clear cache
if st.sidebar.button("🔄 Refresh Model"):
    st.cache_resource.clear()
    st.sidebar.success("✅ Cache cleared! Model will reload.")

selected_stock_name = st.sidebar.selectbox(
    "Select Company",
    list(all_stocks.keys()),
    format_func=lambda x: f"{x} ({all_stocks[x]})"
)
selected_stock = all_stocks[selected_stock_name]

st.sidebar.markdown("---")
selected_period = st.sidebar.selectbox(
    "Time Period",
    ["1d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
    index=4
)

st.sidebar.markdown("---")
st.sidebar.caption("Chart Settings")
candlestick_ma = st.sidebar.checkbox("Show Moving Averages", value=True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("Show RSI", value=False)
show_prediction = st.sidebar.checkbox("Show Price Prediction", value=True)

# Function to fetch stock data with retries and error handling, updated for 1-hour interval
@st.cache_data(ttl=600)
def fetch_stock_data(symbol, period):
    retry_count = 3
    for _ in range(retry_count):
        try:
            stock = yf.Ticker(symbol)
            if period == "1h":
                df = stock.history(period="1d", interval="1m")
                if df.empty:
                    st.warning("No data found for the last 1 hour. Trying with broader period.")
                    # Try fetching data with 5-minute intervals if 1m doesn't work
                    df = stock.history(period="1d", interval="5m")

            else:
                df = stock.history(period=period)  # Fetch data for other periods (daily, weekly, etc.)
            info = stock.info
            return df, info
        except Exception as e:
            st.warning(f"Error fetching data (attempting retry): {e}")
            sleep(randint(1, 3))  # Adding random delay before retry
    st.error("Failed to fetch stock data after multiple attempts.")
    return None, None

# Modified news filtering using GNews API with urllib.request
def get_relevant_news(stock_name, ticker):
    gnews_api_key = os.getenv("NEWS_API_KEY")  # GNews API key from environment
    if not gnews_api_key:
        st.warning("GNEWS_API_KEY not found. Using mock news data.")
        return get_mock_news(stock_name, ticker)
    
    full_name = stock_name
    query = f'"{full_name}" OR "{ticker}"'
    # URL encode the query
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    
    # Add a date filter: only fetch articles from the last 7 days
    from_date = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
    
    url = (
        f"https://gnews.io/api/v4/search?q={encoded_query}"
        f"&lang=en&country=us&max=10&from={from_date}&apikey={gnews_api_key}"
    )
    
    try:
        import json, urllib.request
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data.get("articles", [])
        
        # Strict relevance filtering
        filtered = []
        for article in articles:
            title = article.get('title', '').lower() if article.get('title') else ""
            desc = article.get('description', '').lower() if article.get('description') else ""
            if any([full_name.lower() in title, ticker.lower() in title, full_name.lower() in desc, ticker.lower() in desc]):
                filtered.append(article)
        
        return filtered[:5]
    
    except Exception as e:
        st.warning(f"GNews API unavailable: {e}. Using mock news data.")
        return get_mock_news(stock_name, ticker)


def get_mock_news(stock_name, ticker):
    """Generate mock news articles for when the News API fails"""
    current_date = datetime.now().strftime("%Y-%m-%d")
    mock_articles = [
        {
            "title": f"{stock_name} shows strong quarterly performance",
            "description": f"The company reported better than expected earnings with significant growth in key segments. Analysts remain positive about {stock_name}'s outlook for the upcoming fiscal year.",
            "url": "#",
            "publishedAt": current_date
        },
        {
            "title": f"Analysts recommend buying {stock_name} shares",
            "description": f"Multiple financial institutions have upgraded their rating for {stock_name}, citing strong fundamentals and positive growth indicators in the current market environment.",
            "url": "#",
            "publishedAt": current_date
        },
        {
            "title": f"{stock_name} announces expansion plans",
            "description": f"The company has unveiled plans to expand its operations into new markets, which could drive significant revenue growth in the coming years.",
            "url": "#",
            "publishedAt": current_date
        },
        {
            "title": f"Market outlook positive for {ticker}",
            "description": f"Economic indicators suggest favorable conditions for companies like {stock_name} in the current fiscal quarter. Industry experts predict continued stability.",
            "url": "#",
            "publishedAt": current_date
        },
        {
            "title": f"{stock_name} focuses on technology innovation",
            "description": f"Recent investments in digital transformation and technology infrastructure are expected to improve {stock_name}'s operational efficiency and competitive position.",
            "url": "#",
            "publishedAt": current_date
        }
    ]
    return mock_articles

# Main App
def main():
    st.title(f"{selected_stock_name} Analysis")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Data Loading
    with st.spinner('Loading market data...'):
        df, info = fetch_stock_data(selected_stock, selected_period)

    if df is None or df.empty:
        st.warning("No data available for the selected stock")
        return

    # Key Metrics Grid
    st.subheader("Key Metrics")
    cols = st.columns(4)
    metrics = [
        ("Current Price", f"₹{df['Close'].iloc[-1]:,.2f}"),
        ("Market Cap", f"₹{info.get('marketCap', 0)/1e7:,.1f} Cr"),
        ("52W High", f"₹{info.get('fiftyTwoWeekHigh', 0):,.2f}"),
        ("52W Low", f"₹{info.get('fiftyTwoWeekLow', 0):,.2f}")
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{label}</h3>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">{value}</p>
            </div>
            """, unsafe_allow_html=True)

    # Add Prediction Section
    if show_prediction:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Price Prediction")
        
        with st.spinner("Generating price prediction..."):
            try:
                # First, get the news articles that will be shown in UI
                news_articles = get_relevant_news(selected_stock_name, selected_stock)
                st.sidebar.markdown(f"""
                <div style="font-size: 0.8rem; color: #888; margin-top: 1rem;">
                Debug info:
                - UI has {len(news_articles)} news articles available
                </div>
                """, unsafe_allow_html=True)
                
                # Then use the same news articles for prediction
                predictor = get_stock_predictor()
                try:
                    # Try with external_news parameter (new version)
                    prediction_result = predictor.predict(selected_stock, external_news=news_articles)
                except TypeError as e:
                    # Fall back to old version without external_news parameter
                    st.sidebar.warning("Using legacy prediction model. Please refresh the model to use news data.")
                    prediction_result = predictor.predict(selected_stock)
                
                if prediction_result:
                    # Use the same current price as displayed in key metrics
                    current_price = df['Close'].iloc[-1]
                    predicted_price = prediction_result['predicted_price']
                    news_count = prediction_result['news_count']
                    
                    # Print debug information
                    st.sidebar.markdown(f"""
                    <div style="font-size: 0.8rem; color: #888;">
                    - Prediction uses {news_count} news articles
                    - Same news articles used for UI and prediction
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Calculate percentage change
                    price_change = ((predicted_price - current_price) / current_price) * 100
                    
                    # Create prediction cards
                    pred_cols = st.columns(3)
                    
                    with pred_cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Current Price</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{current_price:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Price</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0;">₹{predicted_price:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with pred_cols[2]:
                        color = "#4CAF50" if price_change >= 0 else "#F44336"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Change</h3>
                            <p style="font-size: 1.5rem; margin: 0.5rem 0; color: {color}">
                                {price_change:+.2f}%
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add prediction confidence indicator with news status
                    has_news_data = news_count > 0
                    confidence_text = "Based on analysis of recent news and market data" if has_news_data else "Based on technical analysis only"
                    confidence_color = "#4CAF50" if has_news_data else "#FFC107"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Prediction Details</h3>
                        <p style="color: {confidence_color}">{confidence_text}</p>
                        <p>Model confidence: {abs(price_change):.1f}%</p>
                        <p style="font-size: 0.9rem; color: #888;">
                            {f"Note: No recent news articles found. Prediction based on historical price patterns." if not has_news_data else f"Based on analysis of {news_count} news articles and market trends."}
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
                        marker=dict(size=10, color='#4FC3F7')
                    ))
                    
                    # Add predicted price point
                    next_date = df.index[-1] + pd.Timedelta(days=1)
                    fig_pred.add_trace(go.Scatter(
                        x=[next_date],
                        y=[predicted_price],
                        mode='markers',
                        name='Predicted Price',
                        marker=dict(size=10, color='#4CAF50' if price_change >= 0 else '#F44336')
                    ))
                    
                    # Add line connecting points
                    fig_pred.add_trace(go.Scatter(
                        x=[df.index[-1], next_date],
                        y=[current_price, predicted_price],
                        mode='lines',
                        name='Price Movement',
                        line=dict(color='#4CAF50' if price_change >= 0 else '#F44336', dash='dash')
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

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Price chart with Bollinger Bands
    st.subheader("Price Movement")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    if candlestick_ma:
        for days, color in [(20, '#FFA726'), (50, '#26C6DA')]:
            ma = df['Close'].rolling(days).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ma,
                name=f'{days} MA',
                line=dict(color=color, width=2)
            ))

    if show_bollinger:
        window = 20
        sma = df['Close'].rolling(window).mean()
        std = df['Close'].rolling(window).std()
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=sma,
            line=dict(color='#FF6F00', width=1.5),
            name='Bollinger Middle (20 SMA)'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=upper_band,
            line=dict(color='#4CAF50', width=1.5),
            name='Upper Band (2σ)',
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=lower_band,
            line=dict(color='#F44336', width=1.5),
            name='Lower Band (2σ)',
            fill='tonexty',
            fillcolor='rgba(76, 175, 80, 0.1)'
        ))

    fig.update_layout(
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI Chart
    if show_rsi:
        def calculate_rsi(data, window=14):
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        rsi = calculate_rsi(df)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Relative Strength Index (RSI)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df.index,
            y=rsi,
            line=dict(color='#8A2BE2', width=2),
            name='RSI'
        ))
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.update_layout(
            height=400,
            template="plotly_dark",
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis_title="RSI"
        )
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Volume Chart
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Trading Volume")
    fig = go.Figure(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker=dict(color='rgba(255, 99, 132, 0.6)'),
        name="Volume"
    ))
    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Display News
    st.subheader("Latest News")
    with st.spinner("Loading news and analyzing sentiment..."):
        news_articles = get_relevant_news(selected_stock_name, selected_stock)
        analyzer = get_finbert_analyzer()
    
    if news_articles:
        for article in news_articles:
            title = article.get('title', '')
            description = article.get('description', '')
            url = article.get('url', '')
            
            # Analyze sentiment using FinBERT
            title_sentiment, title_confidence = analyzer.analyze_sentiment(title)
            desc_sentiment, desc_confidence = analyzer.analyze_sentiment(description)
            
            # Determine overall sentiment (weighted by confidence)
            overall_confidence = (title_confidence + desc_confidence) / 2
            
            # Map sentiment to colors
            sentiment_colors = {
                "positive": "#4CAF50",  # Green
                "negative": "#F44336",  # Red
                "neutral": "#FFC107"    # Yellow
            }
            
            # Use the sentiment with higher confidence
            final_sentiment = title_sentiment if title_confidence > desc_confidence else desc_sentiment
            sentiment_color = sentiment_colors[final_sentiment]
            
            st.markdown(f"""
            <div class="news-card">
                <h3><a href="{url}" target="_blank">{title}</a></h3>
                <p>{description}</p>
                <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                    <span style="background-color: {sentiment_color}; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.9rem;">
                        Sentiment: {final_sentiment.title()} (Confidence: {overall_confidence:.2%})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No news found for the selected stock.")

if __name__ == "__main__":
    main()
