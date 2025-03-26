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
import aiohttp
import asyncio
import yfinance as yf
from datetime import datetime, timedelta
#from time import time, sleep

# Create a cloudscraper session
scraper = cloudscraper.create_scraper()

# Constants
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
]
BASE_URL = "https://www.nseindia.com"
STORED_TICKERS_PATH = "stored_tickers.csv"
ALERTS_FILE = "alerts.json"
CONFIG_FILE = "config.json"
TICKER_PATH = "tickers.csv"

# Headers mimicking your browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/equity-derivatives-watch",
}

# Initial cookie setup
print("Visiting homepage...")
response = scraper.get("https://www.nseindia.com/", headers=headers)
if response.status_code != 200:
    print(f"Failed to load homepage: {response.status_code}")
    exit()

print("Visiting derivatives page...")
scraper.get("https://www.nseindia.com/market-data/equity-derivatives-watch", headers=headers)
time.sleep(2)

# Load/Save Telegram Config and Auto-Scan Settings
def load_config() -> Dict:
    default_config = {
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "auto_scan_enabled": False,
        "auto_scan_interval": 15,
        "enable_telegram_alerts": True,
        "proximity_to_resistance": 1.0  # Add default proximity value
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            return config
    return default_config

def save_config(config: Dict):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)  # indent for readability

# Telegram Integration
async def send_telegram_message(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        st.error("Telegram Bot Token or Chat ID is missing. Please configure them in the sidebar.")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            response_text = await response.text()
            if response.status != 200:
                st.error(f"Failed to send Telegram message: {response_text}")
                if response.status == 404:
                    st.warning("Check your Bot Token and Chat ID. They might be invalid or the bot might not be added to the chat.")
            else:
                st.success(f"Telegram message sent successfully: {message}")

# ... (other unchanged functions like fetch_options_data, process_option_data, etc.) ...
def get_alert_template(recommendation: Dict, ticker: str, expiry: str, underlying: float = None) -> str:
    template = (
        "*SELL CALL ALERT NEW*\n"
        f"Stock: *{ticker}*\n"
        f"Strike: *{recommendation['Strike']:.2f}*\n"
        f"Expiry: *{expiry}*\n"
        f"Premium: *₹{recommendation['Premium']:.2f}*\n"
    )
    if underlying is not None:
        template += f"Underlying: *₹{underlying:.2f}*\n"
    if "Risk_Reward" in recommendation:
        template += f"Risk/Reward: *{recommendation['Risk_Reward']:.2f}*\n"
    template += f"Reason: *{recommendation['Reason']}*"
    return template

# Existing Functions (unchanged unless specified)
def fetch_options_data(symbol: str, _refresh_key: float) -> Optional[Dict]:
    url = f"{BASE_URL}/api/option-chain-equities?symbol={symbol}"
    print(f"Fetching data from: {url}")
    response = scraper.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed with status code: {response.status_code}")
        return None

def process_option_data(data: Dict, expiry: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not data or 'records' not in data or 'data' not in data['records']:
        print("Invalid data, using fallback")
        return get_fallback_data()
    
    options = [item for item in data['records']['data'] if item.get('expiryDate') == expiry]
    if not options:
        print(f"No options found for expiry {expiry}, using fallback")
        return get_fallback_data()
    
    strikes = sorted({item['strikePrice'] for item in options})
    if not strikes:
        print("No strikes found, using fallback")
        return get_fallback_data()
    
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
    
    call_df = pd.DataFrame([{'Strike': k, **v} for k, v in call_data.items()])
    put_df = pd.DataFrame([{'Strike': k, **v} for k, v in put_data.items()])
    
    return call_df, put_df

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
def load_fno_tickers() -> List[str]:
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

#@st.cache_data
def load_tickers() -> List[str]:
    try:
        if os.path.exists(STORED_TICKERS_PATH):
            df = pd.read_csv(STORED_TICKERS_PATH)
            if 'SYMBOL' not in df.columns:
                st.error("Stored CSV file must contain 'SYMBOL' column")
                return ["HDFCBANK"]
            tickers = df['SYMBOL'].dropna().tolist()            
            return tickers if tickers else ["HDFCBANK"]
        else:
            return ["HDFCBANK"]  # Default if no file exists
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        return ["HDFCBANK"]

def calculate_pcr(call_df: pd.DataFrame, put_df: pd.DataFrame) -> float:
    return put_df['OI'].sum() / call_df['OI'].sum() if call_df['OI'].sum() > 0 else 0

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
        if alert_ticker is None or alert_ticker != current_ticker:
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

def generate_call_selling_recommendations(call_df: pd.DataFrame, put_df: pd.DataFrame, underlying: float, max_pain: float, 
                                         pcr: float, support_strike: float, resistance_strike: float, risk_tolerance: float, 
                                         oi_threshold: float, days_to_expiry: float, implied_volatility: float, 
                                         risk_free_rate: float, lot_size: float) -> Tuple[List[Dict], Dict]:
    recommendations = []
    strikes = call_df['Strike'].values
    
    otm_calls = call_df[call_df['Strike'] > underlying].copy()
    if otm_calls.empty:
        return ([{"Strike": None, "Suggestion": "No OTM calls available", "Reason": "Underlying price exceeds all strikes"}], {})

    T = days_to_expiry / 365.0
    sigma = implied_volatility / 100.0
    r = risk_free_rate / 100.0
    
    for strike in otm_calls['Strike']:
        greeks = calculate_option_greeks(S=underlying, K=strike, T=T, r=r, sigma=sigma, option_type="call")
        otm_calls.loc[otm_calls['Strike'] == strike, 'Theta'] = greeks['Theta']
    
    otm_calls['Premium'] = otm_calls['LTP']
    otm_calls['Distance_from_Resistance'] = otm_calls['Strike'] - resistance_strike if resistance_strike else 0
    otm_calls['Risk_Reward'] = np.where(
        (otm_calls['Strike'] - underlying) > 0,
        otm_calls['Premium'] / (otm_calls['Strike'] - underlying),
        0
    )
    
    for index, row in otm_calls.iterrows():
        strike = row['Strike']
        premium = row['Premium']
        oi = row['OI']
        distance = row['Distance_from_Resistance']
        theta = row['Theta']
        risk_reward = row['Risk_Reward']
        
        if abs(row['Change_in_OI']) < oi_threshold and theta < -0.1 and oi > 2000:
            suggestion = "Sell"
            reason = "High OI and favorable Theta decay"
        elif risk_reward > 0.2:
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
    
    recommendations.sort(key=lambda x: (x['Theta'], -x['OI']))
    top_pick = recommendations[0] if recommendations else {}
    
    return recommendations, top_pick

def generate_smart_trade_suggestions(tickers: List[str], expiry: str, bot_token: str, chat_id: str, proximity_percent: float) -> List[Dict]:
    suggestions = []
    refresh_key = time.time()
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                print(f"Failed to fetch data for {ticker}")
                continue
            
            call_df, put_df = process_option_data(data, expiry)
            underlying = data['records'].get('underlyingValue', 0)
            support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
            
            if resistance_strike is None:
                continue
            
            proximity_threshold = resistance_strike * (abs(proximity_percent) / 100)
            distance_to_resistance = resistance_strike - underlying
            
            if proximity_percent >= 0:
                if 0 <= distance_to_resistance <= proximity_threshold:
                    otm_calls = call_df[call_df['Strike'] > underlying]
                    if otm_calls.empty:
                        continue
                    nearest_strike = otm_calls['Strike'].iloc[0]
                    premium = otm_calls[otm_calls['Strike'] == nearest_strike]['LTP'].iloc[0]
                    
                    suggestion = {
                        "Ticker": ticker,
                        "Underlying": underlying,
                        "Strike": nearest_strike,
                        "Premium": premium,
                        "Resistance": resistance_strike,
                        "Distance_to_Resistance": distance_to_resistance,
                        "Reason": f"Underlying within {proximity_percent}% of or at resistance",
                        "Suggestion": "Sell Call"
                    }
                    suggestions.append(suggestion)
            else:
                if -proximity_threshold <= distance_to_resistance <= 0:
                    otm_calls = call_df[call_df['Strike'] > underlying]
                    if otm_calls.empty:
                        continue
                    nearest_strike = otm_calls['Strike'].iloc[0]
                    premium = otm_calls[otm_calls['Strike'] == nearest_strike]['LTP'].iloc[0]
                    
                    suggestion = {
                        "Ticker": ticker,
                        "Underlying": underlying,
                        "Strike": nearest_strike,
                        "Premium": premium,
                        "Resistance": resistance_strike,
                        "Distance_to_Resistance": distance_to_resistance,
                        "Reason": f"Underlying within {proximity_percent}% above resistance",
                        "Suggestion": "Sell Call"
                    }
                    suggestions.append(suggestion)
    
    return suggestions

def get_yesterday_open_price(ticker: str, current_underlying: float) -> float:
    try:
        print(ticker)
        print(current_underlying)
        ticker_ns = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        stock = yf.Ticker(ticker_ns)
        data = stock.history(period="1d")
        
        if data.empty or len(data) < 2:
            print(f"Insufficient or no data for {ticker_ns}, using current underlying as fallback.")
            return current_underlying
        print(data)
        yesterday_open = data['Open'].iloc[-2]
        if pd.isna(yesterday_open):
            print(f"No valid yesterday's open price for {ticker_ns}, using current underlying.")
            return current_underlying

        return yesterday_open
    except Exception as e:
        print(f"Error fetching data for {ticker_ns}: {e}")
        return current_underlying

def generate_lost_momentum_suggestions(tickers: List[str], expiry: str, bot_token: str, chat_id: str) -> List[Dict]:
    suggestions = []
    refresh_key = time.time()
    
    try:
        tickers_df = pd.read_csv(STORED_TICKERS_PATH)
        if 'SYMBOL' not in tickers_df.columns or 'YESTERDAY_OPEN_PRICE' not in tickers_df.columns:
            print(f"Error: {STORED_TICKERS_PATH} must contain 'SYMBOL' and 'YESTERDAY_OPEN_PRICE' columns")
            return suggestions
        ticker_open_prices = dict(zip(tickers_df['SYMBOL'], tickers_df['YESTERDAY_OPEN_PRICE']))
    except Exception as e:
        print(f"Error loading {STORED_TICKERS_PATH}: {e}")
        return suggestions
    
    for ticker in tickers:
        with st.spinner(f"Fetching data for {ticker}..."):
            data = fetch_options_data(ticker, refresh_key)
            if not data or 'records' not in data:
                print(f"Failed to fetch data for {ticker}")
                continue
            
            call_df, put_df = process_option_data(data, expiry)
            if 'Strike' not in call_df.columns:
                print(f"No 'Strike' column in call_df for {ticker}, skipping")
                continue
            
            underlying = data['records'].get('underlyingValue', 0)
            support_strike, resistance_strike = identify_support_resistance(call_df, put_df)
            
            yesterday_open = ticker_open_prices.get(ticker, None)
            if yesterday_open is None:
                print(f"No YESTERDAY_OPEN_PRICE found for {ticker}, skipping")
                continue
            
            try:
                yesterday_open = float(yesterday_open)
            except (ValueError, TypeError):
                print(f"Invalid YESTERDAY_OPEN_PRICE for {ticker}: {yesterday_open}, skipping")
                continue
            
            if yesterday_open > underlying:
                otm_calls = call_df[call_df['Strike'] > underlying]
                if otm_calls.empty:
                    print(f"No OTM calls available for {ticker}")
                    continue
                nearest_strike = otm_calls['Strike'].iloc[0]
                premium = otm_calls[otm_calls['Strike'] == nearest_strike]['LTP'].iloc[0]
                
                suggestion = {
                    "Ticker": ticker,
                    "Yesterday_Open": yesterday_open,
                    "Underlying": underlying,
                    "Strike": nearest_strike,
                    "Premium": premium,
                    "Reason": f"Lost momentum: Yesterday Open ({yesterday_open:.2f}) > Current ({underlying:.2f})",
                    "Suggestion": "Sell Call"
                }
                suggestions.append(suggestion)
                #print(f"Suggestion generated for {ticker}: {suggestion}")
    
    return suggestions

def check_and_trigger_auto_scan():
    current_time = time.time()
    auto_scan_interval_seconds = st.session_state['telegram_config']['auto_scan_interval'] * 60
    
    time_since_last_scan = current_time - st.session_state['last_scan_time']
    print(f"Checking auto-scan: current_time={current_time}, last_scan_time={st.session_state['last_scan_time']}, "
          f"interval={auto_scan_interval_seconds}, enabled={st.session_state['telegram_config']['auto_scan_enabled']}, "
          f"time_since_last_scan={time_since_last_scan}")
    
    if (st.session_state['telegram_config']['auto_scan_enabled'] and 
        time_since_last_scan >= auto_scan_interval_seconds):
        print("Auto-scan triggered!")
        
        # Ensure expiry is set
        if 'expiry' not in st.session_state or not st.session_state['expiry']:
            data = fetch_options_data(st.session_state['ticker'], time.time())
            if data and 'records' in data and 'expiryDates' in data['records']:
                st.session_state['expiry'] = data['records']['expiryDates'][0]
            else:
                print("Failed to set default expiry")
                return
        
        # Set flags to trigger scans in the tabs
        st.session_state['scan_trades_trigger'] = True
        st.session_state['scan_momentum_trigger'] = True
        
        # Update last_scan_time
        st.session_state['last_scan_time'] = current_time
        print(f"Updated last_scan_time to {st.session_state['last_scan_time']}")
        st.rerun()  # Force a rerun to update the UI
    else:
        print(f"Auto-scan not triggered: time_since_last_scan={time_since_last_scan} < interval={auto_scan_interval_seconds}")

# Add this function near other helper functions
# Corrected handle_auto_refresh function
def handle_auto_refresh():
    """Handle auto-refresh logic independently."""
    auto_refresh = st.session_state.get('auto_refresh_checkbox', False)
    if not auto_refresh:
        return
    
    current_time = time.time()  # Use time() if imported as from time import time, otherwise time.time()
    last_refresh = st.session_state.get('refresh_key', time.time())  # Same here
    refresh_interval = 30  # 30 seconds as per checkbox label
    
    if current_time - last_refresh >= refresh_interval:
        st.session_state['refresh_key'] = current_time
        st.session_state['last_refresh_time_display'] = datetime.now().strftime("%H:%M:%S")
        st.rerun()       

def main():
    st.set_page_config(page_title="Options Chain Analysis", layout="wide")
    st.title("Options Chain Analysis")
    
    # Load Telegram Config
    config = load_config()
    if 'telegram_config' not in st.session_state:
        st.session_state['telegram_config'] = config

    # Initialize Session State
    if 'last_scan_time' not in st.session_state:
        st.session_state['last_scan_time'] = time.time()
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
    if 'screener_suggestions' not in st.session_state:
        st.session_state['screener_suggestions'] = None
    if 'momentum_suggestions' not in st.session_state:
        st.session_state['momentum_suggestions'] = None
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = 'HDFCBANK'
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()
    if 'last_refresh_time_display' not in st.session_state:
        st.session_state['last_refresh_time_display'] = "Not refreshed yet"    
    if 'scan_trades_trigger' not in st.session_state:
        st.session_state['scan_trades_trigger'] = False
    if 'scan_momentum_trigger' not in st.session_state:
        st.session_state['scan_momentum_trigger'] = False

    # Check and trigger auto-scan
    check_and_trigger_auto_scan()
    
    # Sidebar Configuration
    with st.sidebar:
        tickers = load_fno_tickers()
        ticker = st.selectbox("Select NSE Ticker:", tickers, 
                             index=tickers.index("HDFCBANK") if "HDFCBANK" in tickers else 0)
        st.session_state['ticker'] = ticker
        auto_refresh = st.checkbox("Auto-Refresh (30s)", key="auto_refresh_checkbox")
        if st.button("Refresh Now"):
            st.session_state['refresh_key'] = time.time()  # Corrected
            st.session_state['last_refresh_time_display'] = datetime.now().strftime("%H:%M:%S")
            st.rerun()
        
        st.write(f"Last Refreshed: {st.session_state['last_refresh_time_display']}")

        st.subheader("Trade Parameters")
        risk_tolerance = st.number_input("Risk Tolerance (₹):", value=5000.0, step=1000.0)
        
        st.subheader("P&L Simulator")
        st.session_state['sold_strike'] = st.number_input("Sold Call Strike:", value=st.session_state['sold_strike'], 
                                                        placeholder="Enter strike", key="sold_strike_input")
        st.session_state['sold_premium'] = st.number_input("Sold Premium:", value=st.session_state['sold_premium'], 
                                                         placeholder="Enter premium", key="sold_premium_input")
        st.session_state['lot_size'] = st.number_input("Lot Size:", value=st.session_state['lot_size'], step=1.0, key="lot_size_input")
        
        st.subheader("Adjustment Inputs")
        oi_threshold = st.number_input("OI Change Threshold:", value=500.0, step=1000.0)

        st.subheader("Greeks Calculator Inputs")
        implied_volatility = st.number_input("Implied Volatility (%):", value=30.0, step=1.0)
        risk_free_rate = st.number_input("Risk-Free Rate (%):", value=5.0, step=0.1)
        days_to_expiry = st.number_input("Days to Expiry:", value=30, step=1)

        st.subheader("Support/Resistance Customization")
        support_color = st.color_picker("Support Line Color:", value="#00FF00")
        resistance_color = st.color_picker("Resistance Line Color:", value="#800080")

        st.subheader("Telegram Integration")
        telegram_bot_token = st.text_input("Telegram Bot Token:", 
                                     value=st.session_state['telegram_config']['telegram_bot_token'], 
                                     type="password")
        telegram_chat_id = st.text_input("Telegram Chat ID:", 
                                   value=st.session_state['telegram_config']['telegram_chat_id'])
        enable_telegram_alerts = st.checkbox("Enable Telegram Alerts", 
                                       value=st.session_state['telegram_config']['enable_telegram_alerts'],
                                       key="telegram_alerts_checkbox")
        
        st.subheader("Automation Settings")
        auto_scan_enabled = st.checkbox("Enable Auto-Scan", 
                                      value=st.session_state['telegram_config']['auto_scan_enabled'],
                                      key="auto_scan_checkbox")
        auto_scan_interval = st.number_input("Auto-Scan Interval (minutes):", 
                                           value=st.session_state['telegram_config']['auto_scan_interval'], 
                                           min_value=1, step=1, key="auto_scan_interval_input")

        # Add Proximity to Resistance input in the sidebar
        proximity_to_resistance = st.number_input(
            "Proximity to Resistance (%):",
            value=st.session_state['telegram_config']['proximity_to_resistance'],
            step=0.2,
            min_value=-10.0,
            max_value=10.0,
            help="Positive: Below resistance; Negative: Above resistance",
            key="proximity_to_resistance_input"
        )

        st.subheader("Upload Ticker CSV")
        uploaded_file = st.file_uploader("Upload CSV with 'SYMBOL' column to replace stored tickers", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'SYMBOL' in df.columns:
                df.to_csv(STORED_TICKERS_PATH, index=False)
                st.success(f"Tickers updated and saved to {STORED_TICKERS_PATH}")
                st.session_state['refresh_key'] = time.time()
            else:
                st.error("Uploaded CSV must contain a 'SYMBOL' column")

        # Update config if changed
        if (st.session_state['telegram_config']['auto_scan_enabled'] != auto_scan_enabled or
            st.session_state['telegram_config']['auto_scan_interval'] != auto_scan_interval):
            st.session_state['telegram_config']['auto_scan_enabled'] = auto_scan_enabled
            st.session_state['telegram_config']['auto_scan_interval'] = auto_scan_interval
            st.session_state['telegram_config']['proximity_to_resistance'] = proximity_to_resistance
            save_config(st.session_state['telegram_config'])

        # Update config if any settings changed
        # config_changed = False
        # if st.session_state['telegram_config']['auto_scan_enabled'] != auto_scan_enabled:
        #     st.session_state['telegram_config']['auto_scan_enabled'] = auto_scan_enabled
        #     config_changed = True
        # if st.session_state['telegram_config']['auto_scan_interval'] != auto_scan_interval:
        #     st.session_state['telegram_config']['auto_scan_interval'] = auto_scan_interval
        #     config_changed = True
        # if st.session_state['telegram_config']['telegram_bot_token'] != telegram_bot_token:
        #     st.session_state['telegram_config']['telegram_bot_token'] = telegram_bot_token
        #     config_changed = True
        # if st.session_state['telegram_config']['telegram_chat_id'] != telegram_chat_id:
        #     st.session_state['telegram_config']['telegram_chat_id'] = telegram_chat_id
        #     config_changed = True
        # if st.session_state['telegram_config']['enable_telegram_alerts'] != enable_telegram_alerts:
        #     st.session_state['telegram_config']['enable_telegram_alerts'] = enable_telegram_alerts
        #     config_changed = True
        # if st.session_state['telegram_config']['proximity_to_resistance'] != proximity_to_resistance:
        #     st.session_state['telegram_config']['proximity_to_resistance'] = proximity_to_resistance
        #     config_changed = True

        if enable_telegram_alerts and (not telegram_bot_token or not telegram_chat_id):
            st.warning("Telegram alerts are enabled but Bot Token or Chat ID is missing. Please configure them.")

    # Data Fetching and Processing
    st.session_state.setdefault('refresh_key', time.time())
    with st.spinner(f"Fetching data for {st.session_state['ticker']}..."):
        data = fetch_options_data(st.session_state['ticker'], st.session_state['refresh_key'])
    
    if not data or 'records' not in data:
        st.error("Failed to load data!")
        return
    
    expiry = st.selectbox("Select Expiry:", data['records']['expiryDates'], index=0)
    st.session_state['expiry'] = expiry

    call_df, put_df = process_option_data(data, expiry)
    underlying = data['records'].get('underlyingValue', 812)
    max_pain = calculate_max_pain(call_df, put_df)
    pcr = calculate_pcr(call_df, put_df)
    support_strike, resistance_strike = identify_support_resistance(call_df, put_df, top_n=3)

    # Check Alerts
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
    
    tabs = st.tabs(["Data", "OI Analysis", "Volume Analysis", "Price Analysis", "P&L Analysis", "Heatmap", "Greeks", "Trade Suggestions", "Trade Screener", "Lost Momentum"])

    # ... (unchanged tabs from 0 to 7) ...
    # Existing Tabs (unchanged until "Trade Screener")
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
            
            y_max = max(pl_values)
            y_min = min(pl_values)
            y_range = y_max - y_min
            y_offset = y_range * 0.05
            
            fig_pl.add_vline(x=max_pain, line_dash="dash", line_color="red", 
                            annotation_text=f"Max Pain ({max_pain:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 3))
            fig_pl.add_vline(x=underlying, line_dash="dash", line_color="blue", 
                            annotation_text=f"Spot ({underlying:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 2))
            if support_strike is not None:
                fig_pl.add_vline(x=support_strike, line_dash="dot", line_color=support_color, 
                                annotation_text=f"Support ({support_strike:.2f})", 
                                annotation_position="top left",
                                annotation=dict(y=y_max + y_offset))
            if resistance_strike is not None:
                fig_pl.add_vline(x=resistance_strike, line_dash="dot", line_color=resistance_color, 
                                annotation_text=f"Resistance ({resistance_strike:.2f})", 
                                annotation_position="top right",
                                annotation=dict(y=y_max))
            breakeven = st.session_state['sold_strike'] + st.session_state['sold_premium']
            fig_pl.add_vline(x=breakeven, line_dash="dash", line_color="orange", 
                            annotation_text=f"Breakeven ({breakeven:.2f})", 
                            annotation_position="top",
                            annotation=dict(y=y_max + y_offset * 4))
            current_pl = st.session_state['sold_premium'] * st.session_state['lot_size'] - max((underlying - st.session_state['sold_strike']), 0) * st.session_state['lot_size']
            fig_pl.add_annotation(x=underlying, y=current_pl, text=f"Current P&L: ₹{current_pl:,.2f}", 
                                 showarrow=True, arrowhead=1)
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

            
            if top_pick:
                st.markdown(
                    f"<div style='background-color: #d4edda; padding: 10px; border-radius: 5px;'>"
                    f"<b>Top Pick:</b> Sell {top_pick['Strike']} Call @ ₹{top_pick['Premium']:.2f} | <b>Reason:</b> {top_pick['Reason']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    with tabs[8]:
        st.subheader("Trade Screener")
        # Use the value from config instead of local variable
        st.write(f"Current Proximity to Resistance: {st.session_state['telegram_config']['proximity_to_resistance']}% "
                 "(configurable in sidebar)")
        scan_button = st.button("Scan Trades")
        
        # Trigger scan if button clicked or auto-scan flag is set
        if scan_button or st.session_state['scan_trades_trigger']:
            screener_tickers = load_tickers()
            if screener_tickers:
                with st.spinner("Scanning trades..."):
                    suggestions = generate_smart_trade_suggestions(
                        screener_tickers, expiry, 
                        telegram_bot_token if enable_telegram_alerts else "", 
                        telegram_chat_id if enable_telegram_alerts else "",
                        st.session_state['telegram_config']['proximity_to_resistance']  # Use config value
                    )
                    st.session_state['screener_suggestions'] = suggestions
                    
                    # Send Telegram message with results
                    if suggestions and st.session_state['telegram_config']['enable_telegram_alerts'] and telegram_bot_token and telegram_chat_id:
                        message = f"*Smart Trade Suggestions (Auto-Scan every {st.session_state['telegram_config']['auto_scan_interval']} min)*\n"
                        for suggestion in suggestions:
                            message += (
                                f"Stock: *{suggestion['Ticker']}*\n"
                                f"Underlying: *₹{suggestion['Underlying']:.2f}*\n"
                                f"Strike: *{suggestion['Strike']:.2f}*\n"
                                f"Premium: *₹{suggestion['Premium']:.2f}*\n"
                                f"Reason: *{suggestion['Reason']}*\n\n"
                            )
                        asyncio.run(send_telegram_message(telegram_bot_token, telegram_chat_id, message))
            else:
                st.warning(f"No tickers found in {STORED_TICKERS_PATH}. Please upload a CSV with 'SYMBOL' column.")
            
            # Reset the trigger after execution
            st.session_state['scan_trades_trigger'] = False
        
        if st.session_state['screener_suggestions'] is not None:
            if st.session_state['screener_suggestions']:
                suggestions_df = pd.DataFrame(st.session_state['screener_suggestions'])
                st.write("### Smart Trade Suggestions")
                styled_df = suggestions_df.style.format({
                    'Underlying': '{:.2f}',
                    'Strike': '{:.2f}',
                    'Premium': '{:.2f}',
                    'Resistance': '{:.2f}',
                    'Distance_to_Resistance': '{:.2f}'
                })
                st.table(styled_df)
            else:
                st.info(f"No smart trade suggestions found based on the {st.session_state['telegram_config']['proximity_to_resistance']}% resistance proximity criteria using stored tickers.")
        else:
            st.info(f"Click 'Scan Trades' or enable auto-scan to analyze tickers from {STORED_TICKERS_PATH}.")

    with tabs[9]:
        st.subheader("Lost Momentum Scanner")
        momentum_scan_button = st.button("Scan Lost Momentum")
        
        # Trigger scan if button clicked or auto-scan flag is set
        if momentum_scan_button or st.session_state['scan_momentum_trigger']:
            momentum_tickers = load_tickers()
            if momentum_tickers:
                with st.spinner("Scanning for lost momentum..."):
                    suggestions = generate_lost_momentum_suggestions(
                        momentum_tickers, expiry, 
                        telegram_bot_token if enable_telegram_alerts else "", 
                        telegram_chat_id if enable_telegram_alerts else ""
                    )
                    st.session_state['momentum_suggestions'] = suggestions
                    
                    # Send Telegram message with results
                    if suggestions and st.session_state['telegram_config']['enable_telegram_alerts'] and telegram_bot_token and telegram_chat_id:
                        message = f"*Lost Momentum Suggestions (Auto-Scan every {st.session_state['telegram_config']['auto_scan_interval']} min)*\n"
                        for suggestion in suggestions:
                            message += (
                                f"Stock: *{suggestion['Ticker']}*\n"
                                f"Yesterday Open: *₹{suggestion['Yesterday_Open']:.2f}*\n"
                                f"Current: *₹{suggestion['Underlying']:.2f}*\n"
                                f"Strike: *{suggestion['Strike']:.2f}*\n"
                                f"Premium: *₹{suggestion['Premium']:.2f}*\n"
                                f"Reason: *{suggestion['Reason']}*\n\n"
                            )
                        asyncio.run(send_telegram_message(telegram_bot_token, telegram_chat_id, message))
            else:
                st.warning(f"No tickers found in {STORED_TICKERS_PATH}. Please upload a CSV with 'SYMBOL' column.")
            
            # Reset the trigger after execution
            st.session_state['scan_momentum_trigger'] = False
        
        if st.session_state['momentum_suggestions'] is not None:
            if st.session_state['momentum_suggestions']:
                suggestions_df = pd.DataFrame(st.session_state['momentum_suggestions'])
                st.write("### Lost Momentum Trade Suggestions")
                styled_df = suggestions_df.style.format({
                    'Yesterday_Open': '{:.2f}',
                    'Underlying': '{:.2f}',
                    'Strike': '{:.2f}',
                    'Premium': '{:.2f}'
                })
                st.table(styled_df)
            else:
                st.info(f"No tickers found with lost momentum using stored tickers from {STORED_TICKERS_PATH}.")
        else:
            st.info(f"Click 'Scan Lost Momentum' or enable auto-scan to analyze tickers from {STORED_TICKERS_PATH}.")

    # Auto-refresh logic without sleep
    auto_refresh = st.session_state.get('auto_refresh_checkbox', False)
    if auto_refresh or st.session_state['telegram_config']['auto_scan_enabled']:
        current_time = time.time()
        if auto_refresh and current_time - st.session_state['refresh_key'] >= 30:
            st.session_state['refresh_key'] = current_time
            st.rerun()

    handle_auto_refresh()        

if __name__ == "__main__":
    main()