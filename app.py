import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta
from time import sleep
from random import randint
from finbert_analyzer import FinBERTAnalyzer

# Initialize FinBERT analyzer
@st.cache_resource
def get_finbert_analyzer():
    return FinBERTAnalyzer()

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


# Improved news filtering
# Improved news filtering using full company name and ticker symbol
def get_relevant_news(stock_name, ticker):
    news_api_key = "813bb17cd2704c12a2acf66732f973bc"  # Replace with your key
    full_name = stock_name
    query = f'"{full_name}" OR "{ticker}"'
    
    # Get news from the last 7 days
    date_from = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        'q': query,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 10,
        'apiKey': news_api_key,
        'from': date_from,
        'qInTitle': stock_name
    }

    try:
        response = requests.get("https://newsapi.org/v2/everything", params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        
        # Strict relevance filtering
        filtered = []
        for article in articles:
            title = article.get('title', '').lower() if article.get('title') else ""
            desc = article.get('description', '').lower() if article.get('description') else ""
            
            # Check if the stock name or ticker is mentioned in the title or description
            if any([full_name.lower() in title, ticker.lower() in title, full_name.lower() in desc, ticker.lower() in desc]):
                filtered.append(article)
        
        return filtered[:5]

    except Exception as e:
        st.error(f"News API Error: {e}")
        return []

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

    # Volume chart and news sections remain the same...

    # Volume Chart
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