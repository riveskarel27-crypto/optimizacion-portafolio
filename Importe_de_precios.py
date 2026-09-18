import yfinance as yf
import pandas as pd

# Cambiar "TICKER" por la acción correspondiente (ej. AAPL, MSFT, etc.)
ticker = "TICKER"
df_raw = yf.download(ticker, start="2023-09-17", end="2026-09-17")
df_raw.to_csv(f"{ticker}.csv")

df_clean = pd.read_csv(f"{ticker}.csv", skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean.set_index('Date', inplace=True)
df_clean.to_csv(f"{ticker}.csv")