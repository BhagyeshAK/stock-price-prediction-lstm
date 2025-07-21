# Stock Price Prediction using LSTM

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 2. Fetch Data
ticker = 'AAPL'  # Change to your preferred stock symbol
data = yf.download(ticker, start='2015-01-01', end='2024-12-31')

# 3. Preprocess Data
data = data[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 4. Create Dataset for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 5. Build and Train LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, y, epochs=10, batch_size=64)

# 6. Predictions
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1, 1))

# 7. Plot Predictions
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# 8. Add Moving Average and RSI Indicators
data['MA20'] = data['Close'].rolling(window=20).mean()
delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -1 * delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# 9. Plot MA and RSI
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(data['Close'], label='Close')
ax[0].plot(data['MA20'], label='MA20', linestyle='--')
ax[0].set_title('Stock Price & 20-Day MA')
ax[0].legend()

ax[1].plot(data['RSI'], color='orange', label='RSI')
ax[1].axhline(70, color='red', linestyle='--')
ax[1].axhline(30, color='green', linestyle='--')
ax[1].set_title('Relative Strength Index (RSI)')
ax[1].legend()

plt.tight_layout()
plt.show()

# 10. Save Model
model.save('lstm_stock_model.h5')
