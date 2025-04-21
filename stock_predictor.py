import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st
import time


def load_and_preprocess_data(filepath, window_size=60):
    # If it's an uploaded file (BytesIO), reset pointer
    if hasattr(filepath, "seek"):
        filepath.seek(0)
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath) # It's a file path (string)
    
    df.dropna(inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, infer_datetime_format=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    df = df[['Date', 'Close']]  # Select date and close price
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Scale the 'Close' price between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])

    # Create sequences of 'window_size' to use as features for LSTM
    X, y = [], []
    for i in range(window_size, len(df_scaled)):
        X.append(df_scaled[i-window_size:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))
    return X_train, X_test, y_train, y_test, scaler, scaled_data


def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))  # Output layer for price prediction

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_and_predict(X_train, X_test, y_train, y_test, model, epochs=2, batch_size=32):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size)
        loss = history.history['loss'][0]
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"🔁 Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")
        time.sleep(0.5)  # Simulate training delay (optional)

    st.success("✅ Model Training Complete!")

    predictions = model.predict(X_test)
    return predictions


def predict_future(model, last_sequence, n_days, scaler):
    future_predictions = []
    current_seq = last_sequence.copy()

    for _ in range(n_days):
        prediction = model.predict(current_seq.reshape(1, -1, 1), verbose=0)
        future_predictions.append(prediction[0, 0])

        # Append prediction to the sequence and slide window
        current_seq = np.append(current_seq[1:], prediction[0, 0])
    
    # Inverse transform to get actual prices
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    return scaler.inverse_transform(future_predictions)
