import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("nvda_lstm_model.keras")
    return model

# Load data
@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Preprocess data
def preprocess_data(data):
    # Rename columns
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    # Calculate moving averages
    data['MA50'] = data['Close_NVDA'].rolling(window=50).mean()
    data['MA200'] = data['Close_NVDA'].rolling(window=200).mean()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    return data

# Make prediction for next day
def predict_next_day(model, data, features, seq_length):
    # Get the last sequence
    scaler = MinMaxScaler(feature_range=(0,1))
    last_sequence = data[features].tail(seq_length)
    last_sequence_scaled = scaler.fit_transform(last_sequence)
    
    # Reshape for LSTM input
    last_sequence_reshaped = last_sequence_scaled.reshape(1, seq_length, len(features))
    
    # Predict
    prediction_scaled = model.predict(last_sequence_reshaped)
    
    # Inverse transform
    temp_array = np.zeros((1, len(features)))
    temp_array[0, 0] = prediction_scaled[0, 0]
    prediction = scaler.inverse_transform(temp_array)[0, 0]
    
    return prediction

# App title
st.title('NVIDIA Stock Price Prediction App')

# Sidebar
st.sidebar.header('User Input Parameters')

# Stock ticker (fixed to NVIDIA)
ticker = 'NVDA'

# Date input
today = datetime.today().date()
five_years_ago = today - timedelta(days=365*5)

start_date = st.sidebar.date_input('Start date', five_years_ago)
end_date = st.sidebar.date_input('End date', today)

# Load model
model = load_trained_model()

# Load and process data
if st.sidebar.button('Fetch Data and Predict'):
    data = load_stock_data(ticker, start_date, end_date)
    
    if data.empty:
        st.error("No data available for the selected date range.")
    else:
        processed_data = preprocess_data(data)
        
        # Show stock price history
        st.subheader('NVIDIA Stock Price History')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['Close_NVDA'],
                                mode='lines',
                                name='Close Price',
                                line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA50'],
                                mode='lines',
                                name='50-day MA',
                                line=dict(color='red')))
        fig.add_trace(go.Scatter(x=processed_data.index, y=processed_data['MA200'],
                                mode='lines',
                                name='200-day MA',
                                line=dict(color='green')))
        
        fig.update_layout(title='NVIDIA Stock Price with Moving Averages',
                        xaxis_title='Date',
                        yaxis_title='Stock Price (USD)',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Features for prediction
        features = ['Open_NVDA', 'MA50', 'MA200']
        
        # Make prediction
        if len(processed_data) >= 60:  # Need at least 60 days for sequence
            next_day_price = predict_next_day(model, processed_data, features, 60)
            
            # Get next trading day
            last_date = processed_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:  # If it's weekend
                next_date = next_date + pd.Timedelta(days=1)  # Skip to next day
            
            # Display prediction
            st.subheader('Stock Price Prediction')
            st.markdown(f"**Predicted stock price for {next_date.date()}**: **${next_day_price:.2f}**")
            
            # Current price
            current_price = processed_data['Close_NVDA'].iloc[-1]
            price_change = next_day_price - current_price
            percent_change = (price_change / current_price) * 100
            
            # Display current price and predicted change
            st.markdown(f"**Current price (as of {last_date.date()})**: **${current_price:.2f}**")
            
            if price_change > 0:
                st.markdown(f"**Predicted Change**: <span style='color:green'>**+${price_change:.2f} (+{percent_change:.2f}%)**</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Predicted Change**: <span style='color:red'>**${price_change:.2f} ({percent_change:.2f}%)**</span>", unsafe_allow_html=True)
            
            # Prediction confidence disclaimer
            st.info("Note: Stock price predictions are estimates based on historical patterns and should not be the sole basis for investment decisions.")
        else:
            st.error("Not enough data available for prediction. Need at least 60 days of data.")

# Add disclaimer at the bottom
st.sidebar.markdown("---")
st.sidebar.caption("This app is for educational purposes only. It does not constitute financial advice.")