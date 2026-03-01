import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import feedparser
import httpx
from typing import List, Tuple, Dict, Any
import warnings

# Suppress yfinance multi-index warning for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

class MarkovChain:
    def __init__(self, symbol: str, bins: int = 5):
        self.symbol = symbol
        self.bins = bins

    def fetch_data(self) -> pd.DataFrame:
        try:
            # Using period="1y" and auto_adjust=True
            data = yf.download(self.symbol, period="1y", interval="1d", progress=False)
            if data.empty:
                return None
            data["returns"] = data["Close"].pct_change()
            return data.dropna()
        except Exception:
            return None

    def build_states(self, data: pd.DataFrame) -> pd.DataFrame:
        data["state"] = pd.qcut(
            data["returns"],
            self.bins,
            labels=False,
            duplicates="drop"
        )
        return data

    def prob_matrix(self) -> Tuple[np.ndarray, int]:
        data = self.fetch_data()
        if data is None:
            return None, None

        data = self.build_states(data)
        states = data["state"].astype(int).values
        
        unique_states = np.unique(states)
        bins = len(unique_states)
        transition_matrix = np.zeros((bins, bins))

        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transition_matrix[current_state][next_state] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        # Fix the np.divide warning by providing 'out' and making it single-line/cleaner
        transition_matrix = np.divide(
            transition_matrix, 
            row_sums, 
            out=np.zeros_like(transition_matrix), 
            where=row_sums != 0
        )

        recent_state = int(states[-1])
        return transition_matrix, recent_state

def add_moving_averages(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    data["SMA_fast"] = data["Close"].rolling(window=fast).mean()
    data["SMA_slow"] = data["Close"].rolling(window=slow).mean()
    return data

def add_crossover_signals(data: pd.DataFrame) -> pd.DataFrame:
    data["signal"] = 0
    data.loc[data["SMA_fast"] > data["SMA_slow"], "signal"] = 1
    data["position_change"] = data["signal"].diff()

    data["event"] = ""
    data.loc[data["position_change"] == 1, "event"] = "Bullish Crossover"
    data.loc[data["position_change"] == -1, "event"] = "Bearish Crossover"
    return data

def fetch_news() -> List[str]:
    feeds = [
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://feeds.reuters.com/reuters/businessNews",
    ]
    headlines = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]: # reduced to 10 per feed for speed
                headlines.append(entry.title)
        except Exception:
            continue
    return list(set(headlines))

def analyze_sentiment(headlines: List[str]) -> List[Tuple[str, float, str]]:
    results = []
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity
        if polarity > 0:
            label = "Positive"
        elif polarity < 0:
            label = "Negative"
        else:
            label = "Neutral"
        results.append((h, polarity, label))
    return sorted(results, key=lambda x: x[1], reverse=True)

def get_market_analysis(symbol: str) -> Dict[str, Any]:
    # Markov Analysis
    mc = MarkovChain(symbol)
    matrix, recent_state = mc.prob_matrix()
    markov_data = None
    if matrix is not None:
        next_probs = matrix[recent_state]
        predicted_state = int(np.argmax(next_probs))
        probability = float(next_probs[predicted_state])
        
        if predicted_state > recent_state:
            bias = "Bullish"
            color = "success"
        elif predicted_state < recent_state:
            bias = "Bearish"
            color = "danger"
        else:
            bias = "Neutral"
            color = "secondary"
        
        markov_data = {
            "current_state": recent_state,
            "predicted_state": predicted_state,
            "probability": f"{probability:.2%}",
            "bias": bias,
            "color": color,
            "state_labels": [
                {"id": 0, "label": "Strong Bearish", "color": "vibrant-red"},
                {"id": 1, "label": "Bearish", "color": "text-muted"},
                {"id": 2, "label": "Neutral", "color": "text-light"},
                {"id": 3, "label": "Bullish", "color": "cyber-blue"},
                {"id": 4, "label": "Strong Bullish", "color": "neon-green"}
            ]
        }

    # Technical Analysis
    data = yf.download(symbol, period="6mo", interval="1d", progress=False)
    tech_data = None
    if not data.empty:
        data = add_moving_averages(data)
        data = add_crossover_signals(data)
        latest = data.iloc[-1]
        
        # In newer yfinance versions, 'Close' can be a series if it's a MultiIndex (even with 1 symbol)
        close_val = float(latest['Close'].iloc[0]) if hasattr(latest['Close'], 'iloc') else float(latest['Close'])
        sma_fast = float(latest['SMA_fast'].iloc[0]) if hasattr(latest['SMA_fast'], 'iloc') else float(latest['SMA_fast'])
        sma_slow = float(latest['SMA_slow'].iloc[0]) if hasattr(latest['SMA_slow'], 'iloc') else float(latest['SMA_slow'])

        if sma_fast > sma_slow:
            trend = "Bullish"
            color = "success"
        else:
            trend = "Bearish"
            color = "danger"
        
        last_event = data[data["event"] != ""]
        last_event_msg = last_event.iloc[-1]["event"] if not last_event.empty else "No recent crossover"

        tech_data = {
            "latest_close": f"${close_val:.2f}",
            "sma_20": f"${sma_fast:.2f}",
            "sma_50": f"${sma_slow:.2f}",
            "trend": trend,
            "color": color,
            "last_event": last_event_msg
        }

    # News Sentiment
    headlines = fetch_news()
    sentiment_results = analyze_sentiment(headlines)
    avg_sentiment = float(np.mean([x[1] for x in sentiment_results])) if sentiment_results else 0.0

    if avg_sentiment > 0:
        overall = "Positive"
        color = "success"
    elif avg_sentiment < 0:
        overall = "Negative"
        color = "danger"
    else:
        overall = "Neutral"
        color = "secondary"

    sentiment_data = {
        "avg_sentiment": f"{avg_sentiment:.2f}",
        "overall": overall,
        "color": color,
        "top_headlines": [{"title": h, "label": l, "polarity": f"{p:.2f}"} for h, p, l in sentiment_results[:5]]
    }

    return {
        "symbol": symbol.upper(),
        "markov": markov_data,
        "tech": tech_data,
        "sentiment": sentiment_data
    }