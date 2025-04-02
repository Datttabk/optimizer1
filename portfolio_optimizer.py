import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns

# Set page configuration
st.set_page_config(
    page_title="AI Portfolio Optimizer",
    layout="centered",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

def inject_css():
    """Inject custom CSS styles"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F5F6FA;
    }
    
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    .header-text {
        font-size: 2.5rem !important;
        color: #1F4173 !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 2rem !important;
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .portfolio-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def get_historical_price(ticker, date):
    """Get historical price with enhanced error handling"""
    try:
        target_date = pd.to_datetime(date).tz_localize('Asia/Kolkata')
        data = yf.Ticker(ticker).history(
            start=target_date - pd.Timedelta(days=7),
            end=target_date + pd.Timedelta(days=1)
        )
        return data[data.index <= target_date].iloc[-1]['Close'] if not data.empty else None
    except Exception as e:
        st.error(f"Price fetch error for {ticker}: {str(e)}")
        return None

def get_historical_data(tickers):
    """Fetch historical data with multiple exchange support and robust validation"""
    try:
        modified_tickers = [t.replace('.NS', '.BO') for t in tickers]
        
        data = yf.download(
            modified_tickers,
            period='3y',
            group_by='ticker',
            progress=False,
            auto_adjust=True
        )
        
        clean_data = pd.DataFrame()
        valid_tickers = []
        
        for t in modified_tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    prices = data.xs('Close', level=1, axis=1)[t]
                else:
                    prices = data['Close']
                
                valid_prices = prices.dropna()
                if len(valid_prices) < 100:
                    st.error(f"⚠️ Insufficient data for {t} ({len(valid_prices)} days found)")
                    continue
                    
                clean_data[t] = valid_prices
                valid_tickers.append(t)
                
            except Exception as e:
                st.error(f"🚨 Could not process {t}: {str(e)}")
                continue

        if clean_data.empty or len(clean_data) < 100:
            st.error("🔴 Optimization requires ≥100 valid trading days")
            return pd.DataFrame()
            
        return clean_data.ffill().bfill().dropna()
        
    except Exception as e:
        st.error(f"🔥 Data download failed: {str(e)}")
        return pd.DataFrame()

def get_live_market_data():
    """Fetch live market data with INR conversions"""
    market_data = {'nifty': np.nan, 'gold': np.nan, 'bitcoin': np.nan}
    try:
        usdinr = yf.Ticker("USDINR=X").history(period='1d')['Close'].iloc[-1]
        
        nifty = yf.Ticker("^NSEI").history(period='1d')
        market_data['nifty'] = nifty['Close'].iloc[-1] if not nifty.empty else np.nan
        
        gold = yf.Ticker("GC=F").history(period='1d')
        market_data['gold'] = gold['Close'].iloc[-1] * usdinr if not gold.empty else np.nan
            
        btc = yf.Ticker("BTC-USD").history(period='1d')
        market_data['bitcoin'] = btc['Close'].iloc[-1] * usdinr if not btc.empty else np.nan
            
    except Exception as e:
        st.error(f"Market data error: {str(e)}")
    return market_data

def calculate_technical_indicators(ticker):
    """Calculate technical indicators with error handling"""
    try:
        data = yf.Ticker(ticker).history(period='6mo')
        if len(data) < 20: return None
            
        data['SMA_50'] = data['Close'].rolling(50).mean()
        data['SMA_200'] = data['Close'].rolling(200).mean()
        
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return {
            'price': data['Close'].iloc[-1],
            'sma50': data['SMA_50'].iloc[-1],
            'sma200': data['SMA_200'].iloc[-1],
            'rsi': data['RSI'].iloc[-1]
        }
    except Exception as e:
        st.error(f"Technical indicator error for {ticker}: {str(e)}")
        return None

def get_market_suitability(ticker):
    """Generate market recommendation with detailed analysis"""
    indicators = calculate_technical_indicators(ticker)
    if not indicators: return "N/A", "gray", "Insufficient data"
    
    score = 0
    reasons = []
    
    price = indicators['price']
    sma50 = indicators['sma50']
    sma200 = indicators['sma200']
    
    if price > sma50:
        score += 1
        reasons.append(f"Price ₹{price:.2f} > 50D MA ₹{sma50:.2f}")
    if sma50 > sma200:
        score += 1
        reasons.append("Golden Cross (50D > 200D MA)")
        
    rsi = indicators['rsi']
    if rsi < 40:
        score += 1
        reasons.append(f"Oversold (RSI {rsi:.1f})")
    elif rsi > 70:
        score -= 1
        reasons.append(f"Overbought (RSI {rsi:.1f})")
    
    if score >= 2: return "Hold", "#00B050", " | ".join(reasons)
    elif score >= 0: return "Neutral", "#FFA500", " | ".join(reasons)
    else: return "Exit", "#FF0000", " | ".join(reasons)

def main():
    inject_css()
    
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

    st.markdown('<p class="header-text">AI Portfolio Optimizer</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Investor Profile")
        investment_budget = st.number_input("Total Investment Budget (₹)", 1000, 1000000000, 100000)
        risk_level = st.select_slider("Risk Tolerance", ['Low', 'Moderate', 'High'])
        timeframe = st.radio("Investment Horizon", ['Short-term (1-3 yrs)', 'Medium-term (3-5 yrs)', 'Long-term (5+ yrs)'])
        market_condition = st.selectbox("Market Outlook", ['Bullish', 'Bearish', 'Volatile'])
        
        st.markdown("---")
        st.header("📥 Add Investment")
        with st.form(key='portfolio_form'):
            st.markdown("**Enter BSE tickers (e.g., TCS.BO, INFY.BO)**")
            ticker_input = st.text_input("Stock Tickers (comma separated)", "TCS.BO, INFY.BO")
            date = st.date_input("Purchase Date", datetime(2020, 1, 1))
            amount = st.number_input("Total Amount (₹)", 1000, 10000000, 100000)
            if st.form_submit_button("Add to Portfolio"):
                raw_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                if not raw_tickers:
                    st.error("Please enter at least one ticker")
                else:
                    amount_per = amount / len(raw_tickers)
                    for t in raw_tickers:
                        price = get_historical_price(t, date)
                        if price and not np.isnan(price):
                            st.session_state.portfolio.append({
                                'ticker': t,
                                'date': date,
                                'amount': amount_per,
                                'purchase_price': price
                            })
                            st.success(f"Added {t} with ₹{amount_per:,.2f}")
                        else:
                            st.error(f"Invalid price for {t} - try different suffix (.BO/.NS)")

    with st.container():
        st.header("📈 Live Market Snapshot")
        market_data = get_live_market_data()
        cols = st.columns(3)
        metrics = [
            ("NIFTY 50", market_data['nifty'], "+1.2%"),
            ("Gold Price", market_data['gold'], "-0.5%"),
            ("Bitcoin", market_data['bitcoin'], "+3.8%")
        ]
        for col, (label, value, delta) in zip(cols, metrics):
            with col:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric(label, 
                        f"₹{value:,.2f}" if not pd.isna(value) else "N/A", 
                        delta=delta)
                st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.header("📊 Portfolio Analysis")
        if not st.session_state.portfolio:
            st.info("💡 Add investments using the sidebar form")
        else:
            holdings_df = pd.DataFrame(st.session_state.portfolio)
            
            current_prices = {}
            for ticker in holdings_df['ticker'].unique():
                try:
                    current_prices[ticker] = yf.Ticker(ticker).history(period='1d')['Close'].iloc[-1]
                except:
                    current_prices[ticker] = np.nan
            
            holdings_df['current_price'] = holdings_df['ticker'].map(current_prices)
            holdings_df['current_value'] = holdings_df['amount'] / holdings_df['purchase_price'] * holdings_df['current_price']
            holdings_df['pct_change'] = (holdings_df['current_price'] / holdings_df['purchase_price'] - 1) * 100
            holdings_df[['recommendation', 'color', 'reason']] = holdings_df['ticker'].apply(
                lambda x: pd.Series(get_market_suitability(x)))
            
            st.subheader("🟢 Market Suitability")
            rec_cols = st.columns(4)
            for idx, row in enumerate(holdings_df.itertuples()):
                with rec_cols[idx % 4]:
                    st.markdown(f"""
                        <div style="background:{row.color}; color:white; 
                            padding:8px 12px; border-radius:20px; 
                            font-size:0.9rem; text-align:center;">
                            {row.ticker}: {row.recommendation}
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
            st.subheader("Current Holdings")
            
            try:
                display_df = holdings_df[[
                    'ticker', 'amount', 'purchase_price',
                    'current_price', 'current_value', 'pct_change',
                    'recommendation', 'reason'
                ]].copy()
                
                display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.2f}")
                display_df['purchase_price'] = display_df['purchase_price'].apply(lambda x: f"₹{x:,.2f}")
                display_df['current_price'] = display_df['current_price'].apply(
                    lambda x: f"₹{x:,.2f}" if not pd.isna(x) else "N/A")
                display_df['current_value'] = display_df['current_value'].apply(
                    lambda x: f"₹{x:,.2f}" if not pd.isna(x) else "N/A")
                display_df['pct_change'] = display_df['pct_change'].apply(
                    lambda x: f"{'↑' if x >= 0 else '↓'} {abs(x):.2f}%")
                
                st.dataframe(
                    display_df,
                    column_config={
                        "recommendation": st.column_config.TextColumn(
                            "Recommendation",
                            help="Market suitability recommendation"
                        ),
                        "reason": st.column_config.TextColumn(
                            "Technical Analysis",
                            help="Detailed technical indicators analysis"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Display error: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
        st.subheader("⚡ Smart Optimization")
        
        try:
            if st.session_state.portfolio:
                tickers = holdings_df['ticker'].unique().tolist()
                data = get_historical_data(tickers)
                
                if not data.empty and len(data) >= 100:
                    mu = expected_returns.capm_return(data)
                    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()
                    
                    # Calculate minimum possible volatility
                    ef_minvol = EfficientFrontier(mu, S)
                    ef_minvol.min_volatility()
                    _, min_vol, _ = ef_minvol.portfolio_performance()
                    
                    # Set target volatility 10% above minimum
                    target_vol = min_vol * 1.10
                    
                    # Perform optimization
                    ef = EfficientFrontier(mu, S)
                    ef.efficient_risk(target_vol)
                    weights = ef.clean_weights()
                    
                    st.write(f"## Optimized Allocation (Target Volatility: {target_vol*100:.1f}%)")
                    cols = st.columns(3)
                    valid_weights = [(k, v) for k, v in weights.items() if v > 0.01]
                    for i, (ticker, weight) in enumerate(valid_weights):
                        with cols[i % 3]:
                            st.metric(
                                label=ticker,
                                value=f"{weight*100:.1f}%",
                                help="Recommended allocation percentage"
                            )
                    
                    ret, vol, sharpe = ef.portfolio_performance()
                    st.write("## Projected Performance")
                    cols = st.columns(3)
                    cols[0].metric("Annual Return", f"{ret*100:.1f}%")
                    cols[1].metric("Annual Volatility", f"{vol*100:.1f}%")
                    cols[2].metric("Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.warning("""
                    Optimization requirements:
                    - Valid historical data for all assets
                    - Minimum 100 trading days of data
                    - At least two assets with sufficient data
                    """)
        
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
