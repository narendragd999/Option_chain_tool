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

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]
BASE_URL = "https://www.nseindia.com"
TICKER_PATH = "E:/apps/Option_Chain_Analyser/tickers.csv"
ALERTS_FILE = "E:/apps/Option_Chain_Analyser/alerts.json"  # File to store alerts persistently

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

# Enhanced Function to Identify Support and Resistance Levels
def identify_support_resistance(call_df: pd.DataFrame, put_df: pd.DataFrame, top_n: int = 3) -> Tuple[float, float]:
    """
    Identify support and resistance levels based on a weighted score of OI and Volume.
    Support: Average strike of top N puts with highest weighted score (OI × Volume).
    Resistance: Average strike of top N calls with highest weighted score (OI × Volume).
    """
    # Calculate weighted score (OI × Volume) for calls and puts
    if not call_df.empty and call_df['OI'].sum() > 0 and call_df['Volume'].sum() > 0:
        call_df['Weighted_Score'] = call_df['OI'] * call_df['Volume']
        # Sort by weighted score and take top N strikes
        top_calls = call_df.nlargest(top_n, 'Weighted_Score')
        # Calculate average strike for resistance
        resistance_strike = top_calls['Strike'].mean()
    else:
        resistance_strike = None

    if not put_df.empty and put_df['OI'].sum() > 0 and put_df['Volume'].sum() > 0:
        put_df['Weighted_Score'] = put_df['OI'] * put_df['Volume']
        # Sort by weighted score and take top N strikes
        top_puts = put_df.nlargest(top_n, 'Weighted_Score')
        # Calculate average strike for support
        support_strike = top_puts['Strike'].mean()
    else:
        support_strike = None
    
    return support_strike, resistance_strike

# Functions for Persistent Alerts
def load_alerts() -> List[Dict]:
    """
    Load alerts from a JSON file.
    """
    if os.path.exists(ALERTS_FILE):
        with open(ALERTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_alerts(alerts: List[Dict]):
    """
    Save alerts to a JSON file.
    """
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f)

# Function to Play Sound Notification
def play_alert_sound():
    """
    Play a sound notification using a JavaScript snippet.
    """
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

# Function to Check Alerts
def check_alerts(alerts: List[Dict], current_ticker: str, underlying: float, call_df: pd.DataFrame, put_df: pd.DataFrame, sold_strike: Optional[float], pcr: float) -> Tuple[List[str], List[int]]:
    """
    Check if any alert conditions are met for the current ticker and return a list of triggered alert messages and indices of one-time alerts to remove.
    """
    triggered_alerts = []
    alerts_to_remove = []
    
    for i, alert in enumerate(alerts):
        # Check if the alert has a ticker key; if not, skip or handle gracefully
        alert_ticker = alert.get('ticker', None)
        if alert_ticker is None:
            # Skip alerts without a ticker or handle them (e.g., assign a default ticker or remove)
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

# Main Application
def main():
    st.set_page_config(page_title="Options Chain Analysis", layout="wide")
    st.title("Options Chain Analysis")
    
    # Initialize Session State for Alerts
    if 'alerts' not in st.session_state:
        st.session_state['alerts'] = load_alerts()  # Load alerts from file
    if 'triggered_alerts' not in st.session_state:
        st.session_state['triggered_alerts'] = []
    if 'last_triggered_count' not in st.session_state:
        st.session_state['last_triggered_count'] = 0

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

        # Color Customization for Support and Resistance Lines
        st.subheader("Support/Resistance Customization")
        support_color = st.color_picker("Support Line Color:", value="#00FF00")  # Default: Green
        resistance_color = st.color_picker("Resistance Line Color:", value="#800080")  # Default: Purple

        # Alerts Section
        st.subheader("Alerts")
        st.write(f"Set alert for ticker: **{ticker}**")
        alert_type = st.selectbox("Alert Type:", ["Spot Price", "OI Change", "PCR"])
        threshold = st.number_input("Threshold Value:", value=0.0, step=0.1)
        direction = st.selectbox("Direction:", ["Above", "Below"])
        one_time = st.checkbox("One-Time Alert", value=False, help="Remove alert after it triggers")
        
        if st.button("Add Alert"):
            alert = {
                "ticker": ticker,  # Associate the alert with the current ticker
                "type": alert_type,
                "threshold": threshold,
                "direction": direction,
                "one_time": one_time
            }
            st.session_state['alerts'].append(alert)
            save_alerts(st.session_state['alerts'])  # Save to file
            st.success(f"Added Alert for {ticker}: {alert_type} {direction} {threshold} {'(One-Time)' if one_time else ''}")

        # Display and Manage Alerts
        if st.session_state['alerts']:
            st.write("**Current Alerts:**")
            for i, alert in enumerate(st.session_state['alerts']):
                # Handle alerts that may not have a 'ticker' key (from older versions)
                alert_ticker = alert.get('ticker', 'Unknown Ticker')
                st.write(f"{i+1}. {alert_ticker}: {alert['type']} {alert['direction']} {alert['threshold']} {'(One-Time)' if alert.get('one_time', False) else ''}")
                if st.button(f"Delete Alert {i+1}", key=f"delete_alert_{i}"):
                    st.session_state['alerts'].pop(i)
                    save_alerts(st.session_state['alerts'])  # Save to file
                    st.rerun()

    # Data Fetching and Processing
    st.session_state.setdefault('refresh_key', time.time())
    with st.spinner(f"Fetching data for {ticker}..."):
        data = fetch_options_data(ticker, st.session_state['refresh_key'])
    
    if not data or 'records' not in data:
        st.error("Failed to load data!")
        return
    
    expiry = st.selectbox("Select Expiry:", data['records']['expiryDates'], index=0)
    call_df, put_df = process_option_data(data, expiry)
    underlying = data['records'].get('underlyingValue', "")
    max_pain = calculate_max_pain(call_df, put_df)
    pcr = calculate_pcr(call_df, put_df)

    # Identify Support and Resistance Levels
    support_strike, resistance_strike = identify_support_resistance(call_df, put_df, top_n=3)

    # Check Alerts for the Current Ticker
    triggered_alerts, alerts_to_remove = check_alerts(st.session_state['alerts'], ticker, underlying, call_df, put_df, sold_strike, pcr)
    
    # Remove one-time alerts that were triggered
    if alerts_to_remove:
        for index in sorted(alerts_to_remove, reverse=True):
            st.session_state['alerts'].pop(index)
        save_alerts(st.session_state['alerts'])  # Save to file

    # Add new triggered alerts and play sound if new alerts are triggered
    if triggered_alerts:
        st.session_state['triggered_alerts'].extend(triggered_alerts)
        # Play sound only if new alerts are triggered
        current_triggered_count = len(st.session_state['triggered_alerts'])
        if current_triggered_count > st.session_state['last_triggered_count']:
            play_alert_sound()
            st.session_state['last_triggered_count'] = current_triggered_count

    # Display Triggered Alerts
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
    
    tabs = st.tabs(["Data", "OI Analysis", "Volume Analysis", "Price Analysis", "P&L Analysis", "P&L/Heatmap", "Greeks Analysis"])
    
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
        # Add Support and Resistance lines with custom colors
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

        # Check if P&L Simulator inputs are provided
        if all(x is not None for x in [sold_strike, sold_premium, lot_size]):
            # --- P&L Graph ---
            st.subheader("P&L Analysis Across Spot Prices")
            # Define the range of spot prices for the graph (e.g., ±500 from the sold strike)
            spot_range = range(int(max(sold_strike - 500, call_df['Strike'].min())), 
                              int(min(sold_strike + 500, call_df['Strike'].max()) + 1), 10)
            # Calculate P&L for each spot price
            pl_values = [sold_premium * lot_size - max((spot - sold_strike), 0) * lot_size for spot in spot_range]
            # Create the P&L graph
            fig_pl = px.line(x=spot_range, y=pl_values, labels={'x': 'Spot Price', 'y': 'P&L (₹)'},
                            title=f'P&L: Sold {sold_strike} Call Option')
            # Add vertical lines for key levels
            fig_pl.add_vline(x=underlying, line_dash="dash", line_color="blue", 
                            annotation_text=f"Spot ({underlying:.2f})", annotation_position="top")
            fig_pl.add_vline(x=max_pain, line_dash="dash", line_color="red", 
                            annotation_text=f"Max Pain ({max_pain:.2f})", annotation_position="top")
            if support_strike is not None:
                fig_pl.add_vline(x=support_strike, line_dash="dot", line_color=support_color, 
                                annotation_text=f"Support ({support_strike:.2f})", annotation_position="top left")
            if resistance_strike is not None:
                fig_pl.add_vline(x=resistance_strike, line_dash="dot", line_color=resistance_color, 
                                annotation_text=f"Resistance ({resistance_strike:.2f})", annotation_position="top right")
            # Calculate breakeven and add a vertical line
            breakeven = sold_strike + sold_premium
            fig_pl.add_vline(x=breakeven, line_dash="dash", line_color="orange", 
                            annotation_text=f"Breakeven ({breakeven:.2f})", annotation_position="top")
            # Add current P&L as an annotation
            current_pl = sold_premium * lot_size - max((underlying - sold_strike), 0) * lot_size
            fig_pl.add_annotation(x=underlying, y=current_pl, text=f"Current P&L: ₹{current_pl:,.2f}", 
                                 showarrow=True, arrowhead=1)
            # Add horizontal line at P&L = 0
            fig_pl.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="P&L = 0", 
                            annotation_position="right")
            st.plotly_chart(fig_pl, use_container_width=True)

            # --- OI Change Table ---
            st.subheader("OI Change Analysis for Nearby Strikes")
            # Select strikes around the sold strike (e.g., ±2 strikes)
            nearby_strikes = sorted(call_df['Strike'].values)
            sold_strike_index = min(range(len(nearby_strikes)), key=lambda i: abs(nearby_strikes[i] - sold_strike))
            start_index = max(0, sold_strike_index - 2)
            end_index = min(len(nearby_strikes), sold_strike_index + 3)
            selected_strikes = nearby_strikes[start_index:end_index]
            # Filter call_df for selected strikes
            oi_data = call_df[call_df['Strike'].isin(selected_strikes)][['Strike', 'OI', 'Change_in_OI']]
            # Add a column to highlight the sold strike
            oi_data['Highlight'] = oi_data['Strike'].apply(lambda x: "Sold Strike" if x == sold_strike else "")
            # Add a column for OI Change interpretation
            oi_data['OI Change Interpretation'] = oi_data['Change_in_OI'].apply(
                lambda x: "Significant Increase" if x > oi_threshold else 
                         "Significant Decrease" if x < -oi_threshold else "Stable"
            )
            st.table(oi_data.style.format({'Strike': '{:.2f}', 'OI': '{:.0f}', 'Change_in_OI': '{:.0f}'}))

            # --- Adjustment Analysis Table ---
            st.subheader("Adjustment Analysis for Sold Call")
            oi_change = call_df[call_df['Strike'] == sold_strike]['Change_in_OI'].iloc[0] if sold_strike in call_df['Strike'].values else 0
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
            st.info("Enter P&L Simulator values (Sold Call Strike, Sold Premium, Lot Size) to see advanced analysis.")    

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
        time.sleep(300)
        st.session_state['refresh_key'] = time.time()
        st.rerun()

if __name__ == "__main__":
    main()