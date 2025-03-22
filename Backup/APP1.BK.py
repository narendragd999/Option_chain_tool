import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time
import random

# Set page configuration
st.set_page_config(page_title="Options Chain Analysis", layout="wide")

# List of User-Agents for rotation
user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Mobile/15E148 Safari/604.1'
]

# Headers with random User-Agent
def get_headers():
    return {
        'User-Agent': random.choice(user_agents),
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/',
    }

# Function to initialize session with cookies
def get_session():
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com/", headers=get_headers(), timeout=10)
        return session
    except requests.exceptions.RequestException as e:
        st.error(f"Error initializing session: {e}")
        return None

# Fetch options chain data with caching
@st.cache_data(ttl=300)  # Cache for 5 minutes, but we'll override with refresh
def fetch_options_chain(symbol, _refresh_key):
    url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
    session = get_session()
    if session:
        try:
            response = session.get(url, headers=get_headers(), timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    return None

# Process data into aligned DataFrames for a specific expiry
def process_data(data, selected_expiry):
    if data and 'records' in data and 'data' in data['records']:
        option_data = data['records']['data']
        
        # Filter data for the selected expiry
        filtered_data = [item for item in option_data if item.get('expiryDate') == selected_expiry]
        
        # Create dictionaries to store CE and PE data with strike prices as keys
        call_dict = {}
        put_dict = {}
        
        for item in filtered_data:
            strike = item['strikePrice']
            if 'CE' in item:
                call_dict[strike] = {
                    'OI': item['CE']['openInterest'],
                    'Change_in_OI': item['CE']['changeinOpenInterest'],
                    'LTP': item['CE']['lastPrice'],
                    'Volume': item['CE']['totalTradedVolume']
                }
            if 'PE' in item:
                put_dict[strike] = {
                    'OI': item['PE']['openInterest'],
                    'Change_in_OI': item['PE']['changeinOpenInterest'],
                    'LTP': item['PE']['lastPrice'],
                    'Volume': item['PE']['totalTradedVolume']
                }
        
        # Get all unique strike prices
        all_strikes = sorted(set(list(call_dict.keys()) + list(put_dict.keys())))
        
        # Build aligned DataFrames
        call_data = []
        put_data = []
        for strike in all_strikes:
            call_entry = call_dict.get(strike, {'OI': 0, 'Change_in_OI': 0, 'LTP': 0, 'Volume': 0})
            put_entry = put_dict.get(strike, {'OI': 0, 'Change_in_OI': 0, 'LTP': 0, 'Volume': 0})
            
            call_data.append({'Strike': strike, **call_entry})
            put_data.append({'Strike': strike, **put_entry})
        
        call_df = pd.DataFrame(call_data)
        put_df = pd.DataFrame(put_data)
        
        return call_df, put_df
    return None, None

# Calculate Max Pain
def calculate_max_pain(call_df, put_df):
    strike_prices = call_df['Strike']
    total_loss = []
    for strike in strike_prices:
        # Loss for calls (if stock closes below strike)
        call_loss = call_df[call_df['Strike'] < strike]['OI'].sum() * (strike - call_df[call_df['Strike'] < strike]['Strike']).sum()
        # Loss for puts (if stock closes above strike)
        put_loss = put_df[put_df['Strike'] > strike]['OI'].sum() * (put_df[put_df['Strike'] > strike]['Strike'] - strike).sum()
        total_loss.append(call_loss + put_loss)
    
    min_loss_idx = total_loss.index(min(total_loss))
    return strike_prices.iloc[min_loss_idx]

# Load ticker symbols from CSV
@st.cache_data
def load_tickers():
    ticker_df = pd.read_csv("D:/apps/Options_chain_tool/tickers.csv")
    return ticker_df['SYMBOL'].tolist()

# Main Streamlit app
def main():
    st.title("Options Chain Analysis")
    
    # Load ticker symbols
    ticker_options = load_tickers()
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        default_ticker = "HDFCBANK"
        ticker = st.selectbox(
            "Select NSE Ticker Symbol:",
            options=ticker_options,
            index=ticker_options.index(default_ticker) if default_ticker in ticker_options else 0
        )
        
        # Refresh controls
        auto_refresh = st.checkbox("Auto-Refresh (every 30s)", value=False)
        if st.button("Refresh Now"):
            st.session_state['refresh_key'] = time.time()
        
        # Alert threshold
        price_change_threshold = st.number_input("Price Change Alert Threshold (%):", min_value=0.0, value=200.0, step=10.0)

    # Initialize refresh key
    if 'refresh_key' not in st.session_state:
        st.session_state['refresh_key'] = time.time()

    if ticker:
        with st.spinner(f"Fetching options chain data for {ticker}..."):
            data = fetch_options_chain(ticker, st.session_state['refresh_key'])
        
        if data and 'records' in data and 'expiryDates' in data['records']:
            # Extract available expiry dates
            expiry_dates = data['records']['expiryDates']
            
            # Dropdown for expiry selection
            selected_expiry = st.selectbox(
                "Select Expiry Date:",
                options=expiry_dates,
                index=0  # Default to the nearest expiry
            )
            
            if selected_expiry:
                call_df, put_df = process_data(data, selected_expiry)
                
                if call_df is not None and put_df is not None:
                    underlying_value = data['records']['underlyingValue']
                    st.subheader(f"Underlying Value for {ticker} (Expiry: {selected_expiry}): {underlying_value}")
                    
                    # Calculate Max Pain
                    max_pain = calculate_max_pain(call_df, put_df)
                    st.metric("Max Pain Strike", max_pain)
                    
                    # Tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Data", "OI Analysis", "Volume Analysis", "Price Analysis"])
                    
                    with tab1:
                        st.subheader("Options Chain Data")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Call Options")
                            st.dataframe(call_df.style.format("{:.2f}", subset=['LTP']))
                        with col2:
                            st.write("Put Options")
                            st.dataframe(put_df.style.format("{:.2f}", subset=['LTP']))
                    
                    with tab2:
                        st.subheader("Open Interest Analysis")
                        # OI comparison chart with Max Pain
                        fig_oi = px.bar(
                            x=call_df['Strike'],
                            y=[call_df['OI'], put_df['OI']],
                            barmode='group',
                            labels={'x': 'Strike Price', 'value': 'Open Interest', 'variable': 'Option Type'},
                            title=f'Call vs Put Open Interest (Expiry: {selected_expiry})',
                            color_discrete_sequence=['#00CC96', '#EF553B']
                        )
                        fig_oi.for_each_trace(lambda t: t.update(name='Call' if t.name == 'wide_variable_0' else 'Put'))
                        fig_oi.add_vline(x=max_pain, line_dash="dash", line_color="red", annotation_text="Max Pain")
                        st.plotly_chart(fig_oi, use_container_width=True)
                        
                        # OI Change chart
                        fig_oi_change = px.bar(
                            x=call_df['Strike'],
                            y=[call_df['Change_in_OI'], put_df['Change_in_OI']],
                            barmode='group',
                            labels={'x': 'Strike Price', 'value': 'Change in OI', 'variable': 'Option Type'},
                            title=f'Change in Open Interest (Expiry: {selected_expiry})',
                            color_discrete_sequence=['#00CC96', '#EF553B']
                        )
                        fig_oi_change.for_each_trace(lambda t: t.update(name='Call' if t.name == 'wide_variable_0' else 'Put'))
                        st.plotly_chart(fig_oi_change, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Volume Analysis")
                        # Volume comparison chart
                        fig_volume = px.bar(
                            x=call_df['Strike'],
                            y=[call_df['Volume'], put_df['Volume']],
                            barmode='group',
                            labels={'x': 'Strike Price', 'value': 'Volume', 'variable': 'Option Type'},
                            title=f'Call vs Put Volume (Expiry: {selected_expiry})',
                            color_discrete_sequence=['#00CC96', '#EF553B']
                        )
                        fig_volume.for_each_trace(lambda t: t.update(name='Call' if t.name == 'wide_variable_0' else 'Put'))
                        st.plotly_chart(fig_volume, use_container_width=True)
                        
                        # LTP comparison chart
                        fig_ltp = px.line(
                            x=call_df['Strike'],
                            y=[call_df['LTP'], put_df['LTP']],
                            labels={'x': 'Strike Price', 'value': 'Last Traded Price', 'variable': 'Option Type'},
                            title=f'Call vs Put LTP (Expiry: {selected_expiry})',
                            color_discrete_sequence=['#00CC96', '#EF553B']
                        )
                        fig_ltp.for_each_trace(lambda t: t.update(name='Call' if t.name == 'wide_variable_0' else 'Put'))
                        st.plotly_chart(fig_ltp, use_container_width=True)
                    
                    with tab4:
                        st.subheader("Price Analysis")
                        # Calculate percentage gain/loss
                        call_df['Gain_Percent'] = ((call_df['LTP'] + underlying_value - call_df['Strike']) / call_df['Strike']) * 100
                        put_df['Gain_Percent'] = ((call_df['Strike'] - underlying_value - put_df['LTP']) / call_df['Strike']) * 100
                        
                        # Alerts
                        call_alerts = call_df[call_df['Gain_Percent'].abs() > price_change_threshold]
                        put_alerts = put_df[put_df['Gain_Percent'].abs() > price_change_threshold]
                        if not call_alerts.empty or not put_alerts.empty:
                            st.warning("Price Change Alerts Triggered!")
                            if not call_alerts.empty:
                                st.write("Call Alerts:", call_alerts[['Strike', 'LTP', 'Gain_Percent']])
                            if not put_alerts.empty:
                                st.write("Put Alerts:", put_alerts[['Strike', 'LTP', 'Gain_Percent']])
                        
                        # Price Gain chart
                        fig_price = px.bar(
                            x=call_df['Strike'],
                            y=[call_df['Gain_Percent'], put_df['Gain_Percent']],
                            barmode='group',
                            labels={'x': 'Strike Price', 'value': 'Gain/Loss %', 'variable': 'Option Type'},
                            title=f'Price Gain/Loss % (Expiry: {selected_expiry})',
                            color_discrete_sequence=['#00CC96', '#EF553B']
                        )
                        fig_price.for_each_trace(lambda t: t.update(name='Call' if t.name == 'wide_variable_0' else 'Put'))
                        st.plotly_chart(fig_price, use_container_width=True)
                
                else:
                    st.error(f"Failed to process options chain data for {ticker} (Expiry: {selected_expiry}).")
            else:
                st.error(f"No expiry date selected for {ticker}.")
        else:
            st.error(f"Failed to load options chain data for {ticker}. This might be due to NSE restrictions or an invalid ticker.")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(30)
            st.session_state['refresh_key'] = time.time()
            st.rerun()

if __name__ == "__main__":
    main()