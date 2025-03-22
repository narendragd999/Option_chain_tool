import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import numpy as np
from scipy.stats import norm
from typing import Tuple, Optional, List, Dict

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]
BASE_URL = "https://www.nseindia.com"
TICKER_PATH = "E:/apps/Option_Chain_Analyser/tickers.csv"

# Utility Functions
def get_headers() -> Dict[str, str]:
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': BASE_URL + "/"
    }

def get_session() -> Optional[requests.Session]:
    session = requests.Session()
    try:
        session.get(BASE_URL, headers=get_headers(), timeout=10)
        return session
    except requests.RequestException as e:
        st.error(f"Session initialization failed: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_options_data(symbol: str, _refresh_key: float) -> Optional[Dict]:
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    session = get_session()
    if session:
        try:
            response = session.get(url, headers=get_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"Data fetch failed for {symbol}: {e}")
    return None

def process_option_data(data: Dict, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data or 'records' not in data or 'data' not in data['records']:
        return get_fallback_data()
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry]
    strikes = sorted({item['strikePrice'] for item in options})
    
    call_data = {s: {'OI': 0, 'Change_in_OI': 0, 'LTP': 0, 'Volume': 0} for s in strikes}
    put_data = {s: {'OI': 0, 'Change_in_OI': 0, 'LTP': 0, 'Volume': 0} for s in strikes}
    
    for item in options:
        strike = item['strikePrice']
        if 'CE' in item:
            call_data[strike] = {k: item['CE'][v] for k, v in 
                               {'OI': 'openInterest', 'Change_in_OI': 'changeinOpenInterest', 
                                'LTP': 'lastPrice', 'Volume': 'totalTradedVolume'}.items()}
        if 'PE' in item:
            put_data[strike] = {k: item['PE'][v] for k, v in 
                              {'OI': 'openInterest', 'Change_in_OI': 'changeinOpenInterest', 
                               'LTP': 'lastPrice', 'Volume': 'totalTradedVolume'}.items()}
    
    return (pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()]),
            pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()]))

def get_fallback_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    strikes = range(700, 861, 20)
    return (pd.DataFrame({
        'Strike': strikes,
        'OI': [200, 300, 400, 500, 600, 700, 800, 1000, 900],
        'Change_in_OI': [50, 60, 70, 80, 90, 100, 200, 600, 300],
        'LTP': [120, 100, 80, 60, 45, 30, 18, 8, 4],
        'Volume': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]
    }), pd.DataFrame({
        'Strike': strikes,
        'OI': [200, 250, 300, 400, 700, 800, 900, 1000, 1200],
        'Change_in_OI': [20, 25, 30, 40, 70, 80, 90, 100, 120],
        'LTP': [3, 5, 8, 12, 18, 25, 40, 60, 80],
        'Volume': [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
    }))

def calculate_max_pain(call_df: pd.DataFrame, put_df: pd.DataFrame) -> float:
    strikes = call_df['Strike']
    losses = [
        sum(max(strike - s, 0) * oi - ltp * oi for s, oi, ltp in zip(call_df['Strike'], call_df['OI'], call_df['LTP'])) +
        sum(max(s - strike, 0) * oi - ltp * oi for s, oi, ltp in zip(put_df['Strike'], put_df['OI'], put_df['LTP']))
        for strike in strikes
    ]
    return strikes[losses.index(min(losses))]

@st.cache_data
def load_tickers() -> List[str]:
    try:
        df = pd.read_csv(TICKER_PATH)
        if 'SYMBOL' not in df.columns:
            st.error("CSV file must contain 'SYMBOL' column")
            return ["HDFCBANK"]
        tickers = df['SYMBOL'].dropna().tolist()
        return tickers if tickers else ["HDFCBANK"]
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return ["HDFCBANK"]

def calculate_pcr(call_df: pd.DataFrame, put_df: pd.DataFrame) -> float:
    return put_df['OI'].sum() / call_df['OI'].sum() if call_df['OI'].sum() > 0 else 0

# Option Greeks Calculator using Black-Scholes Model
def calculate_option_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> Dict[str, float]:
    """
    Calculate Option Greeks using the Black-Scholes model.
    S: Current stock price (underlying)
    K: Strike price
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual)
    sigma: Implied volatility (annual)
    option_type: "call" or "put"
    """
    # Avoid division by zero
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Greeks based on option type
    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:  # put
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    # Common Greeks for both call and put
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 4),
        "Theta": round(theta, 4),
        "Vega": round(vega, 4),
        "Rho": round(rho, 4)
    }

# Function to identify Support and Resistance Levels
def identify_support_resistance(call_df: pd.DataFrame, put_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Identify support and resistance levels based on OI data.
    Support: Strike with the highest OI among puts.
    Resistance: Strike with the highest OI among calls.
    """
    # Find the strike with the highest OI for calls (resistance)
    resistance_strike = call_df.loc[call_df['OI'].idxmax()]['Strike'] if not call_df.empty and call_df['OI'].sum() > 0 else None
    # Find the strike with the highest OI for puts (support)
    support_strike = put_df.loc[put_df['OI'].idxmax()]['Strike'] if not put_df.empty and put_df['OI'].sum() > 0 else None
    
    return support_strike, resistance_strike

# Main Application
def main():
    st.set_page_config(page_title="Options Chain Analysis", layout="wide")
    st.title("Options Chain Analysis")
    
    # Sidebar Configuration
    with st.sidebar:
        tickers = load_tickers()
        ticker = st.selectbox("Select NSE Ticker:", tickers, 
                            index=tickers.index("HDFCBANK") if "HDFCBANK" in tickers else 0)
        auto_refresh = st.checkbox("Auto-Refresh (30s)")
        if st.button("Refresh Now"):
            st.session_state['refresh_key'] = time.time()
        
        price_threshold = st.number_input("Price Change Threshold (%):", 0.0, value=200.0, step=10.0)
        
        st.subheader("P&L Simulator")
        sold_strike = st.number_input("Sold Call Strike:", value=None, placeholder="Enter strike")
        sold_premium = st.number_input("Sold Premium:", value=None, placeholder="Enter premium")
        lot_size = st.number_input("Lot Size:", value=None, placeholder="Enter lot size")
        
        st.subheader("Adjustment Inputs")
        risk_tolerance = st.number_input("Risk Tolerance (₹):", value=5000.0, step=1000.0,
                                       help="Maximum loss you're willing to accept before taking action")
        oi_threshold = st.number_input("OI Change Threshold:", value=500.0, step=100.0)

        # Inputs for Greeks Calculator
        st.subheader("Greeks Calculator Inputs")
        implied_volatility = st.number_input("Implied Volatility (%):", value=30.0, step=1.0, help="Annualized implied volatility in percentage")
        risk_free_rate = st.number_input("Risk-Free Rate (%):", value=5.0, step=0.1, help="Annualized risk-free interest rate in percentage")
        days_to_expiry = st.number_input("Days to Expiry:", value=30, step=1, help="Number of days until option expiry")

    # Data Fetching and Processing
    st.session_state.setdefault('refresh_key', time.time())
    with st.spinner(f"Fetching data for {ticker}..."):
        data = fetch_options_data(ticker, st.session_state['refresh_key'])
    
    if not data or 'records' not in data:
        st.error("Failed to load data!")
        return
    
    expiry = st.selectbox("Select Expiry:", data['records']['expiryDates'], index=0)
    call_df, put_df = process_option_data(data, expiry)
    underlying = data['records'].get('underlyingValue', 812)
    max_pain = calculate_max_pain(call_df, put_df)

    # Identify Support and Resistance Levels
    support_strike, resistance_strike = identify_support_resistance(call_df, put_df)

    # Main Display
    st.subheader(f"Underlying: {underlying:.2f}")
    st.metric("Max Pain Strike", f"{max_pain:.2f}")
    
    tabs = st.tabs(["Data", "OI Analysis", "Volume Analysis", "Price Analysis", "Advanced Tools", "New Features", "Greeks Analysis"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        col1.subheader("Call Options")
        col1.dataframe(call_df.style.format("{:.2f}", subset=['LTP']))
        col2.subheader("Put Options")
        col2.dataframe(put_df.style.format("{:.2f}", subset=['LTP']))

    with tabs[1]:
        fig_oi = px.bar(x=call_df['Strike'], y=[call_df['OI'], put_df['OI']], barmode='group',
                       labels={'x': 'Strike', 'value': 'OI'}, title=f"OI ({expiry})",
                       color_discrete_sequence=['#00CC96', '#EF553B'])
        fig_oi.for_each_trace(lambda t: t.update(name=['Call', 'Put'][int(t.name[-1])]))
        # Add Max Pain line
        fig_oi.add_vline(x=max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain", annotation_position="top")
        # Add Support and Resistance lines
        if support_strike is not None:
            fig_oi.add_vline(x=support_strike, line_dash="dot", line_color="green", annotation_text="Support", annotation_position="top left")
        if resistance_strike is not None:
            fig_oi.add_vline(x=resistance_strike, line_dash="dot", line_color="purple", annotation_text="Resistance", annotation_position="top right")
        st.plotly_chart(fig_oi, use_container_width=True)

    with tabs[2]:
        fig_vol = px.bar(x=call_df['Strike'], y=[call_df['Volume'], put_df['Volume']], barmode='group',
                        labels={'x': 'Strike', 'value': 'Volume'}, title=f"Volume ({expiry})",
                        color_discrete_sequence=['#00CC96', '#EF553B'])
        fig_vol.for_each_trace(lambda t: t.update(name=['Call', 'Put'][int(t.name[-1])]))
        st.plotly_chart(fig_vol, use_container_width=True)

    with tabs[3]:
        call_df['Gain%'] = ((call_df['LTP'] + underlying - call_df['Strike']) / call_df['Strike']) * 100
        put_df['Gain%'] = ((call_df['Strike'] - underlying - put_df['LTP']) / call_df['Strike']) * 100
        fig_price = px.bar(x=call_df['Strike'], y=[call_df['Gain%'], put_df['Gain%']], barmode='group',
                         labels={'x': 'Strike', 'value': 'Gain/Loss %'}, title=f"Price Analysis ({expiry})",
                         color_discrete_sequence=['#00CC96', '#EF553B'])
        fig_price.for_each_trace(lambda t: t.update(name=['Call', 'Put'][int(t.name[-1])]))
        st.plotly_chart(fig_price, use_container_width=True)

    with tabs[5]:
        # OI Heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=[call_df['OI'], put_df['OI']], x=call_df['Strike'], y=['Call', 'Put'],
            colorscale='Viridis'))
        fig_heatmap.update_layout(title=f'OI Heatmap ({expiry})')
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # P&L Simulator
        if all(x is not None for x in [sold_strike, sold_premium, lot_size]):
            spots = range(int(call_df['Strike'].min()), int(call_df['Strike'].max()) + 1, 10)
            pl = [sold_premium * lot_size - max((spot - sold_strike), 0) * lot_size for spot in spots]
            fig_pl = px.line(x=spots, y=pl, labels={'x': 'Spot', 'y': 'P&L (₹)'},
                           title=f'P&L: Sold {sold_strike} Call')
            fig_pl.add_vline(x=underlying, line_dash="dash", line_color="blue", annotation_text="Spot")
            fig_pl.add_vline(x=max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain")
            # Add current P&L as annotation
            current_pl = sold_premium * lot_size - max((underlying - sold_strike), 0) * lot_size
            fig_pl.add_annotation(x=underlying, y=current_pl, text=f"P&L: ₹{current_pl:,.2f}", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_pl, use_container_width=True)

        # Adjustment Table
        st.subheader("Adjustment Analysis")
        if all(x is not None for x in [sold_strike, sold_premium, lot_size]):
            oi_change = call_df[call_df['Strike'] == sold_strike]['Change_in_OI'].iloc[0] if sold_strike in call_df['Strike'].values else 0
            breakeven = sold_strike + sold_premium
            # Calculate P&L: Positive value means profit, negative means loss
            pl_value = sold_premium * lot_size - max((underlying - sold_strike), 0) * lot_size
            
            adjustments = pd.DataFrame({
                'Metric': ['Spot', 'Max Pain', 'Breakeven', 'OI Change', 'Profit/Loss'],
                'Value': [underlying, max_pain, breakeven, oi_change, pl_value],
                'Action': [
                    "Hold" if underlying < max_pain else "Hedge" if underlying > sold_strike else "Monitor",
                    "Hold" if max_pain < sold_strike else "Monitor",
                    "Exit" if underlying > breakeven else "Hold",
                    "Exit" if abs(oi_change) > oi_threshold and oi_change > 0 else "Monitor" if abs(oi_change) > oi_threshold else "Hold",
                    "Hedge" if pl_value < -risk_tolerance else "Hold"
                ],
                'Reason': [
                    f"{'Below' if underlying < max_pain else 'Above' if underlying > sold_strike else 'Between'} key levels",
                    f"Max Pain {'below' if max_pain < sold_strike else 'near/above'} strike",
                    f"Spot {'above' if underlying > breakeven else 'below'} breakeven",
                    f"OI {'↑' if oi_change > 0 else '↓'} {abs(oi_change):.0f}",
                    f"{'Profit' if pl_value >= 0 else 'Loss'} ₹{abs(pl_value):,.0f} vs ₹{risk_tolerance:,.0f}"
                ]
            })
            st.table(adjustments.style.format({'Value': '{:.2f}'}))
        else:
            st.info("Enter P&L Simulator values to see adjustment analysis")

        # PCR Analysis
        st.subheader("Put-Call Ratio (PCR) Analysis")
        pcr = calculate_pcr(call_df, put_df)
        st.metric("PCR", f"{pcr:.2f}")
        st.write("PCR > 1: Bearish sentiment | PCR < 1: Bullish sentiment | PCR ≈ 1: Neutral")

    # New Feature: Greeks Analysis
    with tabs[6]:
        st.subheader("Greeks Analysis")
        if all(x is not None for x in [sold_strike, sold_premium, lot_size]):
            # Calculate time to expiry in years
            T = days_to_expiry / 365.0
            # Convert inputs to decimals
            sigma = implied_volatility / 100.0
            r = risk_free_rate / 100.0
            # Calculate Greeks
            greeks = calculate_option_greeks(
                S=underlying,
                K=sold_strike,
                T=T,
                r=r,
                sigma=sigma,
                option_type="call"
            )
            # Display Greeks in a table
            greeks_df = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Theta", "Vega", "Rho"],
                "Value": [greeks["Delta"], greeks["Gamma"], greeks["Theta"], greeks["Vega"], greeks["Rho"]],
                "Description": [
                    "Rate of change of option price with respect to underlying price",
                    "Rate of change of Delta with respect to underlying price",
                    "Rate of change of option price with respect to time (per day)",
                    "Rate of change of option price with respect to volatility (per 1% change)",
                    "Rate of change of option price with respect to interest rate (per 1% change)"
                ]
            })
            st.table(greeks_df)
        else:
            st.info("Enter P&L Simulator values to see Greeks analysis")

    if auto_refresh:
        time.sleep(30)
        st.session_state['refresh_key'] = time.time()
        st.rerun()

if __name__ == "__main__":
    main()