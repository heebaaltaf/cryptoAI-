
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import time
import requests
import pandas as pd
from threading import Thread, Lock, Event

from datetime import datetime
from ta.momentum import (
    RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator, TSIIndicator
)
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator, IchimokuIndicator
)
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

import numpy as np
import logging



# import spacy
import os
import io
import re
from dotenv import load_dotenv
import boto3
from openai import OpenAI
from pydantic import BaseModel
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if config.env exists (for local testing)
if os.path.exists("config.env"):
    load_dotenv("config.env")
    print("Loaded environment variables from config.env (local testing).")
else:
    print("Using Vercel environment variables.")

# Load environment variables
# load_dotenv("config.env")
OPENAI_API_KEY = os.getenv("open_ai_key")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Check your config.env file.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()



@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Serve the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
allow_origins=["http://127.0.0.1:8000", "https://crypto-ai-pi.vercel.app"]
# # Initialize AWS Polly client
# session = boto3.Session(profile_name="default")  # Replace with your AWS profile
# sts_client = session.client("sts")

# assumed_role = sts_client.assume_role(
#     RoleArn="arn:aws:iam::888577066858:role/lambda_er",  # Replace with your role ARN
#     RoleSessionName="lambdaSession"
# )

# credentials = assumed_role["Credentials"]

# polly_client = boto3.client(
#     "polly",
#     aws_access_key_id=credentials["AccessKeyId"],
#     aws_secret_access_key=credentials["SecretAccessKey"],
#     aws_session_token=credentials["SessionToken"],
# )


# session = boto3.Session(profile_name="default")  # Replace with your AWS profile

polly_client = boto3.client(
    "polly",
    aws_access_key_id=os.getenv("my_AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("my_AWS_SECRET_ACCESS_KEY"),
    region_name="us-east-1"
)

# Session management
sessions = {}




import requests
import pandas as pd
from datetime import datetime
from binance.client import Client
API_KEY =  os.getenv("binance_api_key")
API_SECRET = os.getenv("binance_secret_key")


client = Client(API_KEY, API_SECRET)

import requests
import pandas as pd
from datetime import datetime, timedelta

def get_historical_data_extended(symbol, interval, start_date, end_date):
    """
    Fetch extended historical cryptocurrency data from Binance API.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT').
        interval (str): Data interval (e.g., '1h', '1d').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        pd.DataFrame: Historical price data.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000  # Maximum data points per request
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad HTTP response
        
        data = response.json()
        if not data:
            break  # Exit if no data is returned
        
        all_data.extend(data)
        current_start = int(data[-1][6]) + 1  # Use the last 'close_time' + 1ms for the next request
    
    # Convert data to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Process and clean data
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    
    return df[["open_time", "open", "high", "low", "close", "volume"]]    


def add_technical_indicators(df):
    """
    Add a comprehensive set of technical indicators to the DataFrame.
    """
    # Make a copy of the DataFrame
    df = df.copy()

    # Momentum Indicators
    df.loc[:, "rsi"] = RSIIndicator(close=df["close"], window=14).rsi()
    df.loc[:, "stoch_k"] = StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).stoch()
    df.loc[:, "williams_r"] = WilliamsRIndicator(
        high=df["high"], low=df["low"], close=df["close"], lbp=14
    ).williams_r()
    df.loc[:, "roc"] = ROCIndicator(close=df["close"], window=12).roc()
    df.loc[:, "tsi"] = TSIIndicator(close=df["close"], window_slow=25, window_fast=13).tsi()

    # Trend Indicators
    df.loc[:, "sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df.loc[:, "ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df.loc[:, "sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()
    df.loc[:, "ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df.loc[:, "sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    df.loc[:, "ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()
    df.loc[:, "adx"] = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df.loc[:, "cci"] = CCIIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).cci()
    ichimoku = IchimokuIndicator(
        high=df["high"], low=df["low"], window1=9, window2=26, window3=52
    )
    df.loc[:, "ichimoku_a"] = ichimoku.ichimoku_a()
    df.loc[:, "ichimoku_b"] = ichimoku.ichimoku_b()

    # Volatility Indicators
    bollinger = BollingerBands(close=df["close"], window=20, window_dev=2)
    df.loc[:, "bollinger_hband"] = bollinger.bollinger_hband()
    df.loc[:, "bollinger_lband"] = bollinger.bollinger_lband()
    df.loc[:, "atr"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    keltner = KeltnerChannel(
        high=df["high"], low=df["low"], close=df["close"], window=20
    )
    df.loc[:, "keltner_hband"] = keltner.keltner_channel_hband()
    df.loc[:, "keltner_lband"] = keltner.keltner_channel_lband()

    # Volume Indicators
    df.loc[:, "obv"] = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    df.loc[:, "mfi"] = calculate_mfi(df)
    df.loc[:, "cmf"] = ChaikinMoneyFlowIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=20
    ).chaikin_money_flow()
    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df.loc[:, "macd"] = macd.macd()
    df.loc[:, "macd_signal"] = macd.macd_signal()

    return df

def calculate_mfi(df, window=14):
    """
    Calculate Money Flow Index (MFI) manually.
    
    Args:
        df (pd.DataFrame): DataFrame with columns: 'high', 'low', 'close', 'volume'.
        window (int): Lookback window for MFI calculation.
    
    Returns:
        pd.Series: MFI values.
    """
    # Step 1: Calculate Typical Price (TP)
    df.loc[:,'typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Step 2: Calculate Raw Money Flow (RMF)
    df.loc[:,'money_flow'] = df['typical_price'] * df['volume']
    
    # Step 3: Positive and Negative Money Flow (Vectorized)
    df.loc[:,'positive_flow'] = (df['typical_price'] > df['typical_price'].shift(1)) * df['money_flow']
    df.loc[:,'negative_flow'] = (df['typical_price'] < df['typical_price'].shift(1)) * df['money_flow']
    
    # Step 4: Money Flow Ratio (MFR)
    positive_flow_sum = df['positive_flow'].rolling(window=window).sum()
    negative_flow_sum = df['negative_flow'].rolling(window=window).sum()
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    
    # Step 5: Money Flow Index (MFI)
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi

# Global control for fetch thread
fetch_running = Event()
fetch_thread = None
all_data = pd.DataFrame()

def fetch_data_continuously(symbol, delay_between_fetches):
    """Continuously fetch data while the flag is set."""
    global all_data
    output_file = "temp.csv"

    while fetch_running.is_set():  # Check if the fetch_running flag is set
        try:
            # Fetch new data
            order_book = client.get_order_book(symbol=symbol)
            order_book_timestamp = pd.Timestamp.now(tz='UTC').floor("s").tz_localize(None)
            bid_price = float(order_book['bids'][0][0])
            bid_qty = float(order_book['bids'][0][1])
            ask_price = float(order_book['asks'][0][0])
            ask_qty = float(order_book['asks'][0][1])

            # Create DataFrame for new data
            orderbook_df = pd.DataFrame([{
                'timestamp': order_book_timestamp,
                'bid_price': bid_price,
                'bid_qty': bid_qty,
                'ask_price': ask_price,
                'ask_qty': ask_qty
            }])

            # Append new data to all_data
            if all_data.empty:
                all_data = orderbook_df
            else:
                all_data = pd.concat([all_data, orderbook_df], ignore_index=True)

            # Save to CSV
            if not os.path.isfile(output_file):
                all_data.to_csv(output_file, index=False)
            else:
                orderbook_df.to_csv(output_file, mode="a", header=False, index=False)

            time.sleep(delay_between_fetches)  # Wait before fetching again
        except Exception as e:
            print(f"Error in fetch_data_continuously: {e}")
            break

class ChatRequest(BaseModel):
    user_id: str
    message: str
    crypto: str  # Add the crypto field


@app.post("/begin_conversation")
async def begin_conversation(request: ChatRequest):
    """Start the conversation and fetch process."""
    logger.info(f"Received request: {request}")
    global fetch_thread

    # Parse the cryptocurrency from the request
    selected_crypto = request.crypto
    print(f"Received request to begin conversation with crypto: {selected_crypto}")

    # Start fetching data for the selected cryptocurrency
    if not fetch_running.is_set():
        fetch_running.set()  # Set the flag
        fetch_thread = Thread(target=fetch_data_continuously, args=(selected_crypto, 1), daemon=True)
        fetch_thread.start()
        print(f"Started fetch thread for {selected_crypto}.")
    else:
        print("Fetch thread is already running.")

    return {"message": f"Conversation started and data fetching initiated for {selected_crypto}."}

@app.post("/end_conversation")
async def end_conversation(user_id: str = "default"):
    """End the conversation and stop fetch process."""
    global fetch_thread

    # Stop fetching data
    if fetch_running.is_set():
        fetch_running.clear()  # Clear the flag
        fetch_thread.join()  # Wait for the thread to finish
        print("Fetch thread stopped.")

    # Reset session
    if user_id in sessions:
        sessions.pop(user_id)
    return {"message": "Conversation ended and data fetching stopped."}




def add_volatility_features(df):
    """
    Add volatility-related features to the dataset.
    """
    # Ensure necessary columns are present
    required_columns = ["high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in the dataset.")

    # Add features using .loc to avoid SettingWithCopyWarning
    df.loc[:, "rolling_std_10"] = df["close"].rolling(window=10).std()
    df.loc[:, "rolling_std_20"] = df["close"].rolling(window=20).std()

    # Bollinger Band Width
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df.loc[:, "bb_width"] = bb.bollinger_hband() - bb.bollinger_lband()

    # Average True Range (ATR)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df.loc[:, "atr"] = atr.average_true_range()

    # Price Rate of Change (ROC)
    df.loc[:, "roc"] = df["close"].pct_change(periods=10)

    # EMA Difference
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df.loc[:, "ema_diff"] = short_ema - long_ema

    # Rolling Mean and Std
    df.loc[:, "rolling_mean_5"] = df["close"].rolling(window=5).mean()
    df.loc[:, "rolling_std_5"] = df["close"].rolling(window=5).std()

    # Add time index and future target
    df.loc[:, "time_index"] = range(len(df))
    df.loc[:, "close_next"] = df["close"].shift(-1)

    return df





# Helper Functions
def clean_text(text: str) -> str:
    """Remove emojis and special characters."""
    emoji_pattern = re.compile(
        "[" u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF" "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    return text


@app.get("/", response_class=HTMLResponse)
async def serve_homepage(request: Request):
    """Serve the homepage."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/speak")
async def speak(request: Request):
    """Generate speech using AWS Polly."""
    data = await request.json()
    text = data.get("text", "").strip()
    print(text)
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        sanitized_text = clean_text(text)
        response = polly_client.synthesize_speech(
            Text=sanitized_text,
            OutputFormat="mp3",
            Engine="generative",
            VoiceId="Ruth",  # Replace with the desired voice
            TextType="text",
        )

        audio_stream = io.BytesIO(response["AudioStream"].read())
        audio_stream.seek(0)

        # Return audio stream
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception as e:
        print(f"Error with AWS Polly: {str(e)}")
        raise HTTPException(status_code=500, detail="AWS Polly error")
from datetime import datetime, timedelta

# Define the start and end date as datetime objects
end_date = "2025-12-09"  # Current UTC time
end_date_dt = datetime.utcnow()
print(type(end_date))
#end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
#end_date_dt = end_date.strftime("%Y-%m-%d")
print((end_date_dt))

start_date = end_date_dt - timedelta(days=3)  # 1 year ago

# Convert to strings in the required format
start_date_str = start_date.strftime("%Y-%m-%d")
print((start_date_str))
end_date_str = end_date
API_KEY =os.getenv("google_api")
SEARCH_ENGINE_ID = os.getenv("google_engine_id")


# Load NLP model
# nlp = spacy.load("en_core_web_sm")
# Use the converted strings in the function


#print("Historical data fetched and saved successfully.")


@app.post("/chat")
async def chat(request: ChatRequest):
    """Handle chat conversation."""
    user_id = request.user_id
    user_input = request.message.strip()
    selected_crypto = request.crypto  # Get the selected cryptocurrency


    if not user_input and not sessions.get(user_id):
        introduction = "Hello! I am a Crypto expert named Crypto AI. I can help you with analysis and insights on cryptocurrency markets and blockchain technology. Feel free to ask me anything related to cryptocurrencies or blockchain technology."
        sessions[user_id] = {"conversation_history": [{"role": "assistant", "content": introduction}]}
        return {"reply": introduction}
    
    historical_data = get_historical_data_extended(selected_crypto, "30m", start_date_str, end_date)
    print(historical_data.shape)
    historical_data_with_indicators = add_technical_indicators(historical_data)
    historical_data_with_indicators = add_volatility_features(historical_data_with_indicators)
    print(historical_data.shape)
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    historical_data= historical_data_with_indicators.to_string()

    print(historical_data[:10])

    ask_bid_data = pd.read_csv("temp.csv")
    print(ask_bid_data.shape)
    ask_bid_data = (ask_bid_data[-1000:]).to_string()
    print(ask_bid_data[:10])
    session = sessions.setdefault(user_id, {"conversation_history": []})
    if user_input:
        session["conversation_history"].append({"role": "user", "content": user_input})
    # Your Google Custom Search API details
    

    # User input
    

    # Process input with NLP
    # doc = nlp(user_input)
    # query = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    query  = user_input

    print(f"Extracted query: '{query}'")

   

    # Perform Google Search
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("items", [])[:3]]
        search_summary = "\n".join(snippets)
    else:
        search_summary = "No results found."

    print(f"Search summary: {search_summary}")    
    try:
     
        system_message = f"""You are a highly knowledgeable and analytical AI expert  named Crypto AI specializing in cryptocurrency markets and blockchain technology. Your role is to provide accurate, insightful, and clear explanations, analysis, and recommendations. You ll also be given latest last 1 days bitcoin price data and which you can use for analysis. You must keep the following guidelines in mind while interacting:

1. **Clarity and Accuracy**:
   - Always explain concepts in a precise and easy-to-understand manner, tailoring complexity to the user's level of expertise (beginner, intermediate, advanced).
   - Use simple analogies when explaining technical concepts to beginners.

2. **Market Expertise**:
   - Stay updated on major cryptocurrencies (e.g., Bitcoin, Ethereum) and altcoins, trends in the market, and trading strategies.
   - Provide actionable insights based on price trends, trading volumes, and on-chain data.
   - Clearly explain market metrics like market capitalization, liquidity, volatility, and risk.

3. **Blockchain Insights**:
   - Dive deep into blockchain architecture, consensus mechanisms (e.g., Proof of Work, Proof of Stake), smart contracts, and decentralized finance (DeFi).
   - Offer real-world applications and use cases of blockchain beyond cryptocurrencies, such as supply chain, healthcare, and gaming.

4. **Risk Awareness and Responsibility**:
   - Always provide balanced views, highlighting risks and uncertainties, especially in the volatile crypto market.
   - Avoid making financial advice. Instead, provide analysis and encourage users to conduct their research.

5. **Global Market Trends**:
   - Stay informed about regulations, policies, and major events shaping the global crypto landscape.
   - Explain the impact of geopolitical events, institutional adoption, and regulatory changes on the crypto ecosystem.

6. **Technical Analysis**:
   - Offer insights into technical chart patterns, indicators (e.g., RSI, MACD), and tools for price forecasting.
   - Explain complex strategies like arbitrage, staking, and yield farming in an accessible manner.

7. **Scams and Security**:
   - Educate users about common scams, security best practices, and the importance of private key management.
Here is the latest last 1-day price data (5-minute interval):
{historical_data}

Here is the order book data:
{ask_bid_data}

Here is the Google summary of your query:
{search_summary}"""
        




        messages = [{"role": "system", "content": system_message}] + session["conversation_history"]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )

        reply = response.choices[0].message.content.strip()
        session["conversation_history"].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong.")


@app.post("/chat_recommendation")
async def chat_recommendation(request: ChatRequest):
    """Handle chat conversation."""
    user_id = request.user_id
    user_input = request.message.strip()
    selected_crypto = request.crypto  # Get the selected cryptocurrency


  
    
    historical_data = get_historical_data_extended(selected_crypto, "30m", start_date_str, end_date)
    print(historical_data.shape)
    historical_data_with_indicators = add_technical_indicators(historical_data)
    historical_data_with_indicators = add_volatility_features(historical_data_with_indicators)
    print(historical_data.shape)
    # Save to CSV
    # historical_data_with_indicators.to_csv("btc_usdt_extended_data.csv", index=False)
    historical_data= historical_data_with_indicators.to_string()

    print(historical_data[:10])

    ask_bid_data = pd.read_csv("temp.csv")
    print(ask_bid_data.shape)
    ask_bid_data = (ask_bid_data[-1000:]).to_string()
    print(ask_bid_data[:10])
    session = sessions.setdefault(user_id, {"conversation_history": []})
    # if user_input:
    #     session["conversation_history"].append({"role": "user", "content": user_input})
    # Your Google Custom Search API details
    

    # User input
    

    # Process input with NLP
    # doc = nlp(user_input)
    # query = " ".join([token.text for token in doc if not token.is_stop and not token.is_punct])
    query  = f"latest sentiment data on {selected_crypto} from the social networks along with the latest news "

    print(f"Extracted query: '{query}'")

   

    # Perform Google Search
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        results = response.json()
        snippets = [item["snippet"] for item in results.get("items", [])[:3]]
        search_summary = "\n".join(snippets)
    else:
        search_summary = "No results found."

    print(f"Search summary: {search_summary}")    
    try:
        system_message = f"""
You are a highly skilled AI Day Trading and Futures Analyst specializing in synthesizing various trading strategies to provide actionable insights for day traders and futures traders. Your expertise includes predicting short-term price movements, identifying entry and exit points, and assessing futures trading opportunities based on market data, technical indicators, sentiment analysis, and historical trends.

Your role is to integrate day trading strategies such as momentum trading, range trading, scalping, and breakout trading while also analyzing futures contracts to identify the best opportunities at a specific timestamp.

AI Day Trading and Futures Instructions:
Input Data Analysis:

Analyze the provided data, including:
Historical price data.
Order book depth and bid/ask volumes.
Technical indicators such as RSI, MACD, Bollinger Bands, and VWAP.
Sentiment trends based on news and social media.
Futures market data, including open interest, implied volatility, and contract expiration dates.
Day Trading Strategies Integration:

Combine key day trading strategies:
Momentum Trading: Identify trends using indicators like RSI, MACD, and ROC.
Range Trading: Highlight support and resistance levels for range-bound assets.
Scalping: Detect quick opportunities for small profits.
Breakout Trading: Identify assets breaking past support or resistance levels.
Reversal and Pullback Trading: Assess opportunities for reversals or temporary retracements.
Futures Trading Analysis:

Analyze futures data for the selected asset:
Identify the most liquid and actively traded contracts.
Use open interest and volume to gauge market sentiment and interest.
Evaluate implied volatility to predict price swings.
Assess contango/backwardation for price expectations.
Select the best futures contract at the specific timestamp for actionable predictions.
Prediction and Recommendations:

Combine insights from day trading strategies and futures analysis.
Provide predictions for:
Spot price movements based on historical and real-time data.
Futures contract movements for near-term opportunities.
Highlight the best trading strategy (day trading or futures) for the specific timestamp and explain why.
Output Format:

Present the analysis and predictions clearly and concisely.
Ensure predictions reference the timestamp in human-readable language and the source of information.
Avoid discussing internal analysis steps; focus solely on actionable insights.
Here is the latest last 1-day price data (5-minute interval):
{historical_data}

Here is the order book data:
{ask_bid_data}

Here is the Google summary of your query:
{search_summary}
        



Example Output:use only format not the content
Data Summary for { selected_crypto}:

Timestamp: December 21, 2024, 12:30 PM EST.
Source: Aggregated from [data sources, e.g., QuantifiedStrategies.com, Google News, futures data provider].
Analysis Period: Last 24 hours.
Day Trading Analysis:

Synthesis: "The analysis of Bitcoin over the past 24 hours shows a bullish trend, with strong support at $51,200 and resistance at $53,500. Volatility is high, with Bollinger Bands widening, and sentiment analysis indicates optimism fueled by positive institutional activity."
Prediction: "Bitcoin is likely to test the resistance at $53,500 within the next 4 hours. A breakout may lead to a price surge toward $55,000."
Day Trading Recommendations:
Entry Point: Buy near $51,200.
Exit Point: Sell near $53,500.
Stop-Loss: Place at $50,800.
Futures Trading Analysis:

Selected Contract: Bitcoin Futures (BTC-USD-DEC24).
Synthesis: "The futures market shows high open interest and implied volatility for the December contract. The futures price is trading at a slight premium to the spot price, indicating bullish sentiment."
Prediction: "The December contract is expected to rise to $54,800, aligning with bullish sentiment in the spot market. A breakout in spot price above $53,500 may further boost the futures price."


Futures Trading Recommendations:
Entry Point: Long position at $53,000.
Exit Point: Close position at $54,800.
Stop-Loss: Place at $52,500.
Best Trading Opportunity (Timestamp-Based Decision):

"At this timestamp, the futures contract offers a better risk-reward ratio due to high open interest and clear bullish sentiment. Traders should consider entering a long position in Bitcoin Futures (BTC-USD-DEC24) at $53,000 for a potential upside to $54,800."
Risks and Uncertainties:

"Potential risks include unexpected regulatory news or a sharp decline in liquidity during midday trading hours. Monitor market sentiment closely for any changes."
Next Steps:

"Track price movements at $51,200 (spot support) and $53,500 (spot resistance) for early signals."
"Watch the futures market for sudden shifts in open interest or implied volatility."

"""





        messages = [{"role": "system", "content": system_message}] # + session["conversation_history"]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.8,
            max_tokens=300,
        )

        reply = response.choices[0].message.content.strip()
        # session["conversation_history"].append({"role": "assistant", "content": reply})
        return {"reply": reply}
    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Something went wrong.")




# @app.post("/reset")
# async def reset(user_id: str = "default"):
#     """Reset conversation history for the given user ID."""
#     if user_id in sessions:
#         sessions.pop(user_id)  # Clear session data
#         return {"message": "Conversation history reset."}
#     return {"message": "No active session found to reset."}

@app.post("/reset")
async def reset(user_id: str = "default"):
    """Reset conversation history and stop fetching."""
    global fetch_running

    # Clear session data
    if user_id in sessions:
        sessions.pop(user_id)

    # Stop the fetch thread
    if fetch_running.is_set():
        fetch_running.clear()  # Clear the flag to stop fetching
        print("Fetch thread stopped.")

    return {"message": "Conversation and data fetching stopped successfully."}
