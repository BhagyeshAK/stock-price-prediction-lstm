# stock-price-prediction-lstm
Predict future stock prices using LSTM and technical indicators (MA, RSI)
📌 Project Description

- Developed a *stock price prediction model* using LSTM (Long Short-Term Memory) networks.
- Used **real historical stock data** from Yahoo Finance via the `yfinance` API.
- Preprocessed and normalized data to prepare for deep learning model training.
- Trained an LSTM model to learn *time-series patterns* and forecast future prices.
- Included technical indicators:
  - *Moving Average (MA20)* – to smooth price trends.
  - *Relative Strength Index (RSI)* – to detect overbought/oversold conditions.
- Visualized actual vs predicted prices for performance comparison.
- Project designed for **educational purposes**, showcasing how ML can be used in financial forecasting.
- (Optional) Ready to deploy as an *interactive Streamlit dashboard*.

 🛠 Tools & Libraries Used
- Python
- Pandas
- NumPy
- scikit-learn
- Keras (TensorFlow backend)
- Matplotlib
- yfinance (Yahoo Finance API)
- (Optional) Streamlit for web deployment


📊 Workflow Overview
1. Fetch Stock Data
2. Preprocess Data
3. Build & Train LSTM Model
4. Save the Model
5. Plot & Evaluate

✅ Result
A trained LSTM model capable of learning time-based stock patterns and forecasting future trends based on previous data.
