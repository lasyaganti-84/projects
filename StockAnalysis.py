import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 

# Step 1: Data Collection
ticker = 'NVDA'
df = yf.download(ticker, start='2020-01-01', end='2023-01-01')

# Step 2: Data Cleaning
df.dropna(inplace=True)

# Step 3: Technical Analysis
# Calculate Moving Averages
df['MA50'] = df['Close'].rolling(window=50).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Calculate Relative Strength Index (RSI)
def calculate_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['Close'], 14)

# Calculate Bollinger Bands
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()
df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

# Step 4: Trend Analysis
# Identify trends based on Moving Averages crossovers
df['Signal'] = 0
df['Signal'][50:] = np.where(df['MA50'][50:] > df['MA200'][50:], 1, 0)
df['Position'] = df['Signal'].diff()

# Step 5: Visualization
plt.figure(figsize=(14,10))

# Plot Closing Price and Moving Averages
plt.subplot(2, 1, 1)
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['MA50'], label='50-Day MA', color='red')
plt.plot(df['MA200'], label='200-Day MA', color='green')
plt.title(f'{ticker} Stock Price and Moving Averages')
plt.legend()

# Plot RSI
plt.subplot(2, 1, 2)
plt.plot(df['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title(f'{ticker} Relative Strength Index (RSI)')
plt.legend()

plt.tight_layout()
plt.show()

# Plot Bollinger Bands
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price', color='blue')
plt.plot(df['Upper_Band'], label='Upper Bollinger Band', color='red')
plt.plot(df['Lower_Band'], label='Lower Bollinger Band', color='green')
plt.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='grey', alpha=0.3)
plt.title(f'{ticker} Bollinger Bands')
plt.legend()
plt.show()

# Step 6: Reporting
# Summary of Findings
print(f"Average closing price: {df['Close'].mean():.2f}")
print(f"Max closing price: {df['Close'].max():.2f}")
print(f"Min closing price: {df['Close'].min():.2f}")
print(f"RSI value on last trading day: {df['RSI'].iloc[-1]:.2f}")
print(f"Moving Average Crossover Signal on last trading day: {df['Signal'].iloc[-1]}")
