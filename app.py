import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

# Title
st.title("BTC-USD Weekly GBM Swing Range")

# Sidebar inputs
st.sidebar.header("Parameters")
t_days = st.sidebar.slider("Swing Horizon (days)", 1, 30, 7)
n_sims = st.sidebar.number_input("Monte Carlo Simulations", 1000, 50000, 10000, step=1000)

# Functions
def fetch_data_binance(symbol='BTCUSDT', interval='1d', limit=1000):
    """
    Fetch daily BTCUSDT klines from Binance public API (max 1000 days).
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        raise ValueError(f"Binance API returned status {resp.status_code}")
    data = resp.json()
    if not data:
        return pd.DataFrame()
    cols = ['openTime','open','high','low','close','volume','closeTime',
            'quoteAssetVolume','numberOfTrades','takerBuyBaseAssetVolume',
            'takerBuyQuoteAssetVolume','ignore']
    df = pd.DataFrame(data, columns=cols)
    df['Date'] = pd.to_datetime(df['openTime'], unit='ms')
    df.set_index('Date', inplace=True)
    df['S'] = df['close'].astype(float)
    return df[['S']]


def fit_gbm(log_returns):
    mu = log_returns.mean() + 0.5 * log_returns.std()**2
    sigma = log_returns.std()
    return mu, sigma


def swing_range(S0, mu, sigma, days, days_per_year=365):
    t = days / days_per_year
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t)
    lower = S0 * np.exp(drift - diffusion)
    upper = S0 * np.exp(drift + diffusion)
    return lower, upper


def simulate_prices(S0, mu, sigma, days, days_per_year=365, n_sims=10000):
    t = days / days_per_year
    drift = (mu - 0.5 * sigma**2) * t
    diffusion = sigma * np.sqrt(t)
    Z = np.random.randn(n_sims)
    return S0 * np.exp(drift + diffusion * Z)

# Load data
st.text("Fetching BTC-USD history via Binance API (up to 1000 days)...")
try:
    df = fetch_data_binance(symbol='BTCUSDT')
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if df.empty:
    st.error("No price data returned. Binance limit or network issue.")
    st.stop()

# Compute daily log returns
df['r'] = np.log(df['S'] / df['S'].shift(1))
df.dropna(inplace=True)

# Fit GBM parameters
mu_d, sigma_d = fit_gbm(df['r'])
# Convert to weekly metrics (7 days)
mu_w = mu_d * 7
sigma_w = sigma_d * np.sqrt(7)
# Display as percentages
st.write(f"**Estimated weekly drift (μ₇)**: {mu_w * 100:.4f}%")
st.write(f"**Estimated weekly volatility (σ₇)**: {sigma_w * 100:.4f}%")

# Latest price
S0 = df['S'].iloc[-1]
st.write(f"**Latest BTC-USD price (S₀)**: ${S0:,.2f}")

# Swing range
lo, hi = swing_range(S0, mu_d, sigma_d, days=t_days)
st.write(f"**1σ {t_days}-day range:** ${lo:,.2f} – ${hi:,.2f}")

# Monte Carlo percentiles
sim_prices = simulate_prices(S0, mu_d, sigma_d, days=t_days, n_sims=n_sims)
p5, p95 = np.percentile(sim_prices, [5, 95])
st.write(f"**Simulated 5th/95th percentiles:** ${p5:,.2f} – ${p95:,.2f}")

# Plot distribution
fig, ax = plt.subplots()
ax.hist(sim_prices, bins=50)
ax.axvline(p5, linestyle='--', label='5th percentile')
ax.axvline(p95, linestyle='--', label='95th percentile')
ax.set_title(f"Distribution of {t_days}-day Simulated Prices")
ax.set_xlabel("Price (USD)")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)