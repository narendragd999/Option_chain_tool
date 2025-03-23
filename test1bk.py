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
import json
import os
from streamlit.components.v1 import html
import cloudscraper
import aiohttp  # For asynchronous HTTP requests to Telegram API
import asyncio

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]
BASE_URL = "https://www.nseindia.com"
TICKER_PATH = "tickers.csv"
ALERTS_FILE = "alerts.json"

# Headers mimicking your browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Step 1: Visit the homepage to get initial cookies
print("Visiting homepage...")
response = scraper.get("https://www.nseindia.com/", headers=headers)
if response.status_code != 200:
    print(f"Failed to load homepage: {response.status_code}")
    exit()

# Step 2: Visit the derivatives page
print("Visiting derivatives page...")
scraper.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
time.sleep(2)

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    """Send a message to Telegram asynchronously."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"  # Allows formatting like bold, italic, etc.
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                st.error(f"Failed to send Telegram message: {await response.text()}")
            else:
                print(f"Telegram message sent successfully: {message}")

def get_alert_template(recommendation: Dict, ticker: str, expiry: str) -> str:
    """Generate a formatted alert message for Telegram."""
    template = (
        "*SELL CALL ALERT*\n"
        f"Stock: *{ticker}*\n"
        f"Strike: *{recommendation['Strike']:.2f}*\n"
        f"Expiry: *{expiry}*\n"
        f"Premium: *₹{recommendation['Premium']:.2f}*\n"
        f"Risk/Reward: *{recommendation['Risk_Reward']:.2f}*\n"
        f"Reason: *{recommendation['Reason']}*"
    )
    return template

def fetch_options_data(symbol: str, _refresh_key: float) -> Optional[Dict]:
    # Step 3: Fetch the option chain data
    #url = "https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    print(f"Fetching data from: {url}")
    response = scraper.get(url, headers=headers)   

    # Check response
    if response.status_code == 200:
        data = response.json()
        print("Success! Data fetched:")
        return response.json()
    else:
        print(f"Failed with status code: {response.status_code}")
        print(f"Response text: {response.text}")
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
    if T <= 0 or sigma <= 0:
        return {"Delta": 0, "Gamma": 0, "Theta": 0, "Vega": 0, "Rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "Delta": round(delta, 4),
        "Gamma": round(gamma, 4),
        "Theta": round(theta, 4),
        "Vega": round(vega, 4),
        "Rho": round(rho, 4)
    }

def identify_support_resistance(call_df: pd.DataFrame, put_df: pd.DataFrame, top_n: int = 3) -> Tuple[float, float]:
    if not call_df.empty and call_df['OI'].sum() > 0 and call_df['Volume'].sum() > 0:
        call_df['Weighted_Score'] = call_df['OI'] * call_df['Volume']
        top_calls = call_df.nlargest(top_n, 'Weighted_Score')
        resistance_strike = top_calls['Strike'].mean()
    else:
        resistance_strike = None

    if not put_df.empty and put_df['OI'].sum() > 0 and put_df['Volume'].sum() > 0:
        put_df['Weighted_Score'] = put_df['OI'] * put_df['Volume']
        top_puts = put_df.nlargest(top_n, 'Weighted_Score')
        support_strike = top_puts['Strike'].mean()
    else:
        support_strike = None
    
    return support_strike, resistance_strike

def load_alerts() -> List[Dict]:
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_alerts(alerts: List[Dict]):
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f)

def play_alert_sound():
    sound_script = """
    <audio id="alertSound" autoplay>
        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    <script>
        document.getElementById('alertSound').play();
    </script>
    """
    html(sound_script)

def check_alerts(alerts: List[Dict], current_ticker: str, underlying: float, call_df: pd.DataFrame, put_df: pd.DataFrame, sold_strike: Optional[float], pcr: float) -> Tuple[List[str], List[int]]:
    triggered_alerts = []
    alerts_to_remove = []
    
    for i, alert in enumerate(alerts):
        alert_ticker = alert.get('ticker', None)
        if alert_ticker is None:
            continue
        if alert_ticker != current_ticker:
            continue
        
        alert_type = alert['type']
        threshold = alert['threshold']
        direction = alert['direction']
        is_one_time = alert.get('one_time', False)
        
        triggered = False
        if alert_type == "Spot Price":
            if direction == "Above" and underlying > threshold:
                triggered_alerts.append(f"Spot Price Alert for {current_ticker}: Underlying ({underlying:.2f}) crossed above {threshold:.2f}")
                triggered = True
            elif direction == "Below" and underlying < threshold:
                triggered_alerts.append(f"Spot Price Alert for {current_ticker}: Underlying ({underlying:.2f}) crossed below {threshold:.2f}")
                triggered = True
        
        elif alert_type == "OI Change" and sold_strike is not None:
            oi_change = call_df[call_df['Strike'] == sold_strike]['Change_in_OI'].iloc[0] if sold_strike in call_df['Strike'].values else 0
            if direction == "Above" and oi_change > threshold:
                triggered_alerts.append(f"OI Change Alert for {current_ticker}: OI Change ({oi_change:.0f}) at strike {sold_strike} exceeded {threshold:.0f}")
                triggered = True
            elif direction == "Below" and oi_change < threshold:
                triggered_alerts.append(f"OI Change Alert for {current_ticker}: OI Change ({oi_change:.0f}) at strike {sold_strike} dropped below {threshold:.0f}")
                triggered = True
        
        elif alert_type == "PCR":
            if direction == "Above" and pcr > threshold:
                triggered_alerts.append(f"PCR Alert for {current_ticker}: PCR ({pcr:.2f}) exceeded {threshold:.2f}")
                triggered = True
            elif direction == "Below" and pcr < threshold:
                triggered_alerts.append(f"PCR Alert for {current_ticker}: PCR ({pcr:.2f}) dropped below {threshold:.2f}")
                triggered = True
        
        if triggered and is_one_time:
            alerts_to_remove.append(i)
    
    return triggered_alerts, alerts_to_remove

# Modified Function for Call Selling Recommendations (Updated Reasons)
def generate_call_selling_recommendations(call_df: pd.DataFrame, put_df: pd.DataFrame, underlying: float, max_pain: float, 
                                         pcr: float, support_strike: float, resistance_strike: float, risk_tolerance: float, 
                                         oi_threshold: float, days_to_expiry: float, implied_volatility: float, 
                                         risk_free_rate: float, lot_size: float) -> Tuple[List[Dict], Dict]:
    """
    Generate call selling recommendations with additional metrics like Theta and Risk-Reward.
    Returns a tuple of (recommendations list, top pick dictionary).
    """
    recommendations = []
    strikes = call_df['Strike'].values
    
    # Filter out-of-the-money (OTM) calls (strike > underlying)
    otm_calls = call_df[call_df['Strike'] > underlying].copy()
    if otm_calls.empty:
        return ([{"Strike": None, "Suggestion": "No OTM calls available", "Reason": "Underlying price exceeds all strikes"}], {})

    # Calculate Greeks for each strike
    T = days_to_expiry / 365.0
    sigma = implied_volatility / 100.0
    r = risk_free_rate / 100.0
    
    for strike in otm_calls['Strike']:
        greeks = calculate_option_greeks(S=underlying, K=strike, T=T, r=r, sigma=sigma, option_type="call")
        otm_calls.loc[otm_calls['Strike'] == strike, 'Theta'] = greeks['Theta']
    
    otm_calls['Premium'] = otm_calls['LTP']
    
    # Calculate additional metrics
    otm_calls['Distance_from_Resistance'] = otm_calls['Strike'] - resistance_strike if resistance_strike else 0
    otm_calls['Risk_Reward'] = np.where(
        (otm_calls['Strike'] - underlying) > 0,
        otm_calls['Premium'] / (otm_calls['Strike'] - underlying),
        0
    )
    
    # Generate recommendations
    for index, row in otm_calls.iterrows():
        strike = row['Strike']
        premium = row['Premium']
        oi = row['OI']
        distance = row['Distance_from_Resistance']
        theta = row['Theta']
        risk_reward = row['Risk_Reward']
        
        # Determine suggestion and reason
        if abs(row['Change_in_OI']) < oi_threshold and theta < -0.1 and oi > 2000:  # High OI and favorable Theta decay
            suggestion = "Sell"
            reason = "High OI and favorable Theta decay"
        elif risk_reward > 0.2:  # Good risk-reward ratio
            suggestion = "Sell"
            reason = "Good risk-reward ratio"
        else:
            suggestion = "Monitor"
            reason = "Neutral conditions"
        
        recommendations.append({
            "Strike": strike,
            "Premium": premium,
            "OI": oi,
            "Distance_from_Resistance": distance,
            "Theta": theta,
            "Risk_Reward": risk_reward,
            "Suggestion": suggestion,
            "Reason": reason,
            "Lot_Size": lot_size
        })
    
    # Sort recommendations by Theta (most negative first) and OI (highest first)
    recommendations.sort(key=lambda x: (x['Theta'], -x['OI']))
    
    # Select top pick (best Theta and high OI)
    top_pick = recommendations[0] if recommendations else {}
    
    return recommendations, top_pick

# Main Application
def main():
    st.set_page_config(page_title="Options Chain Analysis", layout="wide")
    st.title("Options Chain Analysis")
    
    # Initialize Session State
    if 'alerts' not in st.session_state:
        st.session_state['alerts'] = load_alerts()
    if 'triggered_alerts' not in st.session_state:
        st.session_state['triggered_alerts'] = []
    if 'last_triggered_count' not in st.session_state:
        st.session_state['last_triggered_count'] = 0
    if 'sold_strike' not in st.session_state:
        st.session_state['sold_strike'] = None
    if 'sold_premium' not in st.session_state:
        st.session_state['sold_premium'] = None
    if 'lot_size' not in st.session_state:
        st.session_state['lot_size'] = 100.0

    # Sidebar Configuration
    with st.sidebar:
        tickers = load_tickers()
        ticker = st.selectbox("Select NSE Ticker:", tickers, 
                            index=tickers.index("HDFCBANK") if "HDFCBANK" in tickers else 0)
        auto_refresh = st.checkbox("Auto-Refresh (30s)")
        if st.button("Refresh Now"):
            st.session_state['refresh_key'] = time.time()
        
        st.subheader("Trade Parameters")
        risk_tolerance = st.number_input("Risk Tolerance (₹):", value=5000.0, step=1000.0)
        st.session_state['lot_size'] = st.number_input("Lot Size:", value=st.session_state['lot_size'], step=1.0, key="lot_size_input")
        
        st.subheader("P&L Simulator")
        st.session_state['sold_strike'] = st.number_input("Sold Call Strike:", value=st.session_state['sold_strike'], 
                                                        placeholder="Enter strike", key="sold_strike_input")
        st.session_state['sold_premium'] = st.number_input("Sold Premium:", value=st.session_state['sold_premium'], 
                                                         placeholder="Enter premium", key="sold_premium_input")
        
        st.subheader("Adjustment Inputs")
        oi_threshold = st.number_input("OI Change Threshold:", value=500.0, step=100.0)

        st.subheader("Greeks Calculator Inputs")
        implied_volatility = st.number_input("Implied Volatility (%):", value=30.0, step=1.0)
        risk_free_rate = st.number_input("Risk-Free Rate (%):", value=5.0, step=0.1)
        days_to_expiry = st.number_input("Days to Expiry:", value=30, step=1)

        st.subheader("Support/Resistance Customization")
        support_color = st.color_picker("Support Line Color:", value="#00FF00")
        resistance_color = st.color_picker("Resistance Line Color:", value="#800080")

        st.subheader("Telegram Integration")
        telegram_bot_token = st.text_input("Telegram Bot Token:", type="password")
        telegram_chat_id = st.text_input("Telegram Chat ID:")
        enable_telegram_alerts = st.checkbox("Enable Telegram Alerts for Trade Suggestions", value=False)

        st.subheader("Alerts")
        # Existing alert configuration remains unchanged...

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
    pcr = calculate_pcr(call_df, put_df)
    support_strike, resistance_strike = identify_support_resistance(call_df, put_df, top_n=3)

    # Check Alerts (unchanged)
    triggered_alerts, alerts_to_remove = check_alerts(st.session_state['alerts'], ticker, underlying, call_df, put_df, 
                                                    st.session_state['sold_strike'], pcr)
    if alerts_to_remove:
        for index in sorted(alerts_to_remove, reverse=True):
            st.session_state['alerts'].pop(index)
        save_alerts(st.session_state['alerts'])

    if triggered_alerts:
        st.session_state['triggered_alerts'].extend(triggered_alerts)
        current_triggered_count = len(st.session_state['triggered_alerts'])
        if current_triggered_count > st.session_state['last_triggered_count']:
            play_alert_sound()
            st.session_state['last_triggered_count'] = current_triggered_count

    if st.session_state['triggered_alerts']:
        st.warning("**Alerts Triggered:**")
        for alert_msg in st.session_state['triggered_alerts']:
            st.write(f"- {alert_msg}")
        if st.button("Clear Alerts"):
            st.session_state['triggered_alerts'] = []
            st.session_state['last_triggered_count'] = 0
            st.rerun()

    # Main Display
    st.subheader(f"Underlying: {underlying:.2f}")
    st.metric("Max Pain Strike", f"{max_pain:.2f}")
    
    tabs = st.tabs(["Data", "OI Analysis", "Volume Analysis", "Price Analysis", "P&L Analysis", "P&L/Heatmap", "Greeks Analysis", "Trade Suggestions"])
    
    # Other tabs remain unchanged until "Trade Suggestions"...

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
        fig_oi.add_vline(x=max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain", annotation_position="top")
        if support_strike is not None:
            fig_oi.add_vline(x=support_strike, line_dash="dot", line_color=support_color, 
                            annotation_text=f"Support ({support_strike:.2f})", annotation_position="top left")
        if resistance_strike is not None:
            fig_oi.add_vline(x=resistance_strike, line_dash="dot", line_color=resistance_color, 
                            annotation_text=f"Resistance ({resistance_strike:.2f})", annotation_position="top right")
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

    with tabs[4]:
        st.subheader("Advanced Options Analysis")
        if all(x is not None for x in [st.session_state['sold_strike'], st.session_state['sold_premium'], st.session_state['lot_size']]):
            st.subheader("P&L Analysis Across Spot Prices")
            spot_range = range(int(max(st.session_state['sold_strike'] - 500, call_df['Strike'].min())), 
                              int(min(st.session_state['sold_strike'] + 500, call_df['Strike'].max()) + 1), 10)
            pl_values = [st.session_state['sold_premium'] * st.session_state['lot_size'] - max((spot - st.session_state['sold_strike']), 0) * st.session_state['lot_size'] for spot in spot_range]
            fig_pl = px.line(x=spot_range, y=pl_values, labels={'x': 'Spot Price', 'y': 'P&L (₹)'},
                            title=f'P&L: Sold {st.session_state["sold_strike"]} Call Option')
            
            # Add vertical lines with non-overlapping annotations
            # Base y position for annotations (top of the chart)
            y_max = max(pl_values)
            y_min = min(pl_values)
            y_range = y_max - y_min
            y_offset = y_range * 0.05  # 5% of the y-range for spacing between annotations
            
            # Add Max Pain line
            fig_pl.add_vline(x=max_pain, line_dash="dash", line_color="red", 
                            annotation_text=f"Max Pain ({max_pain:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 3))  # Highest position
            
            # Add Spot line
            fig_pl.add_vline(x=underlying, line_dash="dash", line_color="blue", 
                            annotation_text=f"Spot ({underlying:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 2))  # Second highest
            
            # Add Support line (if available)
            if support_strike is not None:
                fig_pl.add_vline(x=support_strike, line_dash="dot", line_color=support_color, 
                                annotation_text=f"Support ({support_strike:.2f})", 
                                annotation_position="top left",
                                annotation=dict(y=y_max + y_offset))  # Third position
            
            # Add Resistance line (if available)
            if resistance_strike is not None:
                fig_pl.add_vline(x=resistance_strike, line_dash="dot", line_color=resistance_color, 
                                annotation_text=f"Resistance ({resistance_strike:.2f})", 
                                annotation_position="top right",
                                annotation=dict(y=y_max))  # Lowest of the top annotations
            
            # Add Breakeven line
            breakeven = st.session_state['sold_strike'] + st.session_state['sold_premium']
            fig_pl.add_vline(x=breakeven, line_dash="dash", line_color="orange", 
                            annotation_text=f"Breakeven ({breakeven:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 4))  # Above Max Pain
            
            # Add Current P&L annotation
            current_pl = st.session_state['sold_premium'] * st.session_state['lot_size'] - max((underlying - st.session_state['sold_strike']), 0) * st.session_state['lot_size']
            fig_pl.add_annotation(x=underlying, y=current_pl, text=f"Current P&L: ₹{current_pl:,.2f}", 
                                 showarrow=True, arrowhead=1)
            
            # Add P&L = 0 line
            fig_pl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="P&L = 0", 
                            annotation_position="right")
            
            st.plotly_chart(fig_pl, use_container_width=True)

            st.subheader("OI Change Analysis for Nearby Strikes")
            nearby_strikes = sorted(call_df['Strike'].values)
            sold_strike_index = min(range(len(nearby_strikes)), key=lambda i: abs(nearby_strikes[i] - st.session_state['sold_strike']))
            start_index = max(0, sold_strike_index - 2)
            end_index = min(len(nearby_strikes), sold_strike_index + 3)
            selected_strikes = nearby_strikes[start_index:end_index]
            oi_data = call_df[call_df['Strike'].isin(selected_strikes)][['Strike', 'OI', 'Change_in_OI']]
            oi_data['Highlight'] = oi_data['Strike'].apply(lambda x: "Sold Strike" if x == st.session_state['sold_strike'] else "")
            oi_data['OI Change Interpretation'] = oi_data['Change_in_OI'].apply(
                lambda x: "Significant Increase" if x > oi_threshold else 
                         "Significant Decrease" if x < -oi_threshold else "Stable"
            )
            st.table(oi_data.style.format({'Strike': '{:.2f}', 'OI': '{:.0f}', 'Change_in_OI': '{:.0f}'}))

            st.subheader("Adjustment Analysis for Sold Call")
            oi_change = call_df[call_df['Strike'] == st.session_state['sold_strike']]['Change_in_OI'].iloc[0] if st.session_state['sold_strike'] in call_df['Strike'].values else 0
            pl_value = st.session_state['sold_premium'] * st.session_state['lot_size'] - max((underlying - st.session_state['sold_strike']), 0) * st.session_state['lot_size']
            
            adjustments = pd.DataFrame({
                'Metric': ['Spot', 'Max Pain', 'Breakeven', 'OI Change', 'Profit/Loss'],
                'Value': [underlying, max_pain, breakeven, oi_change, pl_value],
                'Action': [
                    "Hold" if underlying < max_pain else "Hedge" if underlying > st.session_state['sold_strike'] else "Monitor",
                    "Hold" if max_pain < st.session_state['sold_strike'] else "Monitor",
                    "Exit" if underlying > breakeven else "Hold",
                    "Exit" if abs(oi_change) > oi_threshold and oi_change > 0 else "Monitor" if abs(oi_change) > oi_threshold else "Hold",
                    "Hedge" if pl_value < -risk_tolerance else "Hold"
                ],
                'Reason': [
                    f"{'Below' if underlying < max_pain else 'Above' if underlying > st.session_state['sold_strike'] else 'Between'} key levels",
                    f"Max Pain {'below' if max_pain < st.session_state['sold_strike'] else 'near/above'} strike",
                    f"Spot {'above' if underlying > breakeven else 'below'} breakeven",
                    f"OI {'↑' if oi_change > 0 else '↓'} {abs(oi_change):.0f}",
                    f"{'Profit' if pl_value >= 0 else 'Loss'} ₹{abs(pl_value):,.0f} vs ₹{risk_tolerance:,.0f}"
                ]
            })
            st.table(adjustments.style.format({'Value': '{:.2f}'}))

        else:
            st.info("Enter P&L Simulator values to see advanced analysis.") 

    with tabs[5]:
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=[call_df['OI'], put_df['OI']], x=call_df['Strike'], y=['Call', 'Put'],
            colorscale='Viridis'))
        fig_heatmap.update_layout(title=f'OI Heatmap ({expiry})')
        st.plotly_chart(fig_heatmap, use_container_width=True)

        if all(x is not None for x in [st.session_state['sold_strike'], st.session_state['sold_premium'], st.session_state['lot_size']]):
            spots = range(int(call_df['Strike'].min()), int(call_df['Strike'].max()) + 1, 10)
            pl = [st.session_state['sold_premium'] * st.session_state['lot_size'] - max((spot - st.session_state['sold_strike']), 0) * st.session_state['lot_size'] for spot in spots]
            fig_pl = px.line(x=spots, y=pl, labels={'x': 'Spot', 'y': 'P&L (₹)'},
                           title=f'P&L: Sold {st.session_state["sold_strike"]} Call')
            fig_pl.add_vline(x=underlying, line_dash="dash", line_color="blue", annotation_text="Spot")
            fig_pl.add_vline(x=max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain")
            current_pl = st.session_state['sold_premium'] * st.session_state['lot_size'] - max((underlying - st.session_state['sold_strike']), 0) * st.session_state['lot_size']
            fig_pl.add_annotation(x=underlying, y=current_pl, text=f"P&L: ₹{current_pl:,.2f}", showarrow=True, arrowhead=1)
            st.plotly_chart(fig_pl, use_container_width=True)

            st.subheader("Adjustment Analysis")
            oi_change = call_df[call_df['Strike'] == st.session_state['sold_strike']]['Change_in_OI'].iloc[0] if st.session_state['sold_strike'] in call_df['Strike'].values else 0
            breakeven = st.session_state['sold_strike'] + st.session_state['sold_premium']
            pl_value = st.session_state['sold_premium'] * st.session_state['lot_size'] - max((underlying - st.session_state['sold_strike']), 0) * st.session_state['lot_size']
            
            adjustments = pd.DataFrame({
                'Metric': ['Spot', 'Max Pain', 'Breakeven', 'OI Change', 'Profit/Loss'],
                'Value': [underlying, max_pain, breakeven, oi_change, pl_value],
                'Action': [
                    "Hold" if underlying < max_pain else "Hedge" if underlying > st.session_state['sold_strike'] else "Monitor",
                    "Hold" if max_pain < st.session_state['sold_strike'] else "Monitor",
                    "Exit" if underlying > breakeven else "Hold",
                    "Exit" if abs(oi_change) > oi_threshold and oi_change > 0 else "Monitor" if abs(oi_change) > oi_threshold else "Hold",
                    "Hedge" if pl_value < -risk_tolerance else "Hold"
                ],
                'Reason': [
                    f"{'Below' if underlying < max_pain else 'Above' if underlying > st.session_state['sold_strike'] else 'Between'} key levels",
                    f"Max Pain {'below' if max_pain < st.session_state['sold_strike'] else 'near/above'} strike",
                    f"Spot {'above' if underlying > breakeven else 'below'} breakeven",
                    f"OI {'↑' if oi_change > 0 else '↓'} {abs(oi_change):.0f}",
                    f"{'Profit' if pl_value >= 0 else 'Loss'} ₹{abs(pl_value):,.0f} vs ₹{risk_tolerance:,.0f}"
                ]
            })
            st.table(adjustments.style.format({'Value': '{:.2f}'}))
        else:
            st.info("Enter P&L Simulator values to see adjustment analysis")

        st.subheader("Put-Call Ratio (PCR) Analysis")
        st.metric("PCR", f"{pcr:.2f}")
        st.write("PCR > 1: Bearish sentiment | PCR < 1: Bullish sentiment | PCR ≈ 1: Neutral")

    with tabs[6]:
        st.subheader("Greeks Analysis")
        if all(x is not None for x in [st.session_state['sold_strike'], st.session_state['sold_premium'], st.session_state['lot_size']]):
            T = days_to_expiry / 365.0
            sigma = implied_volatility / 100.0
            r = risk_free_rate / 100.0
            greeks = calculate_option_greeks(
                S=underlying,
                K=st.session_state['sold_strike'],
                T=T,
                r=r,
                sigma=sigma,
                option_type="call"
            )
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

    with tabs[7]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Underlying", f"{underlying:.2f}")
        col2.metric("PCR", f"{pcr:.2f}")
        col3.metric("Max Pain", f"{max_pain:.2f}")
        col4.metric("Expiry", expiry)

        recommendations, top_pick = generate_call_selling_recommendations(
            call_df, put_df, underlying, max_pain, pcr, support_strike, resistance_strike, 
            risk_tolerance, oi_threshold, days_to_expiry, implied_volatility, risk_free_rate, 
            st.session_state['lot_size']
        )
        
        if recommendations[0]["Strike"] is None:
            st.warning("No favorable call selling opportunities found.")
        else:
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df = recommendations_df[['Strike', 'Premium', 'OI', 'Distance_from_Resistance', 
                                                    'Theta', 'Risk_Reward', 'Suggestion', 'Reason']]
            
            st.write("### Recommendations Table")
            styled_df = recommendations_df.style.format({
                'Strike': '{:.2f}',
                'Premium': '{:.2f}',
                'OI': '{:.0f}',
                'Distance_from_Resistance': '{:.2f}',
                'Theta': '{:.4f}',
                'Risk_Reward': '{:.4f}'
            })
            st.table(styled_df)

            # Send Telegram Alert for Top Pick
            if top_pick and enable_telegram_alerts and telegram_bot_token and telegram_chat_id:
                if 'last_top_pick' not in st.session_state or st.session_state['last_top_pick'] != top_pick:
                    alert_message = get_alert_template(top_pick, ticker, expiry)
                    asyncio.run(send_telegram_message(telegram_bot_token, telegram_chat_id, alert_message))
                    st.session_state['last_top_pick'] = top_pick
                    st.success("Telegram alert sent for top pick!")

            if top_pick:
                st.markdown(
                    f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>"
                    f"<b>Top Pick:</b> Sell {top_pick['Strike']} Call @ ₹{top_pick['Premium']:.2f} | <b>Reason:</b> {top_pick['Reason']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    if auto_refresh:
        time.sleep(30)
        st.session_state['refresh_key'] = time.time()
        st.rerun()

if __name__ == "__main__":
    main()