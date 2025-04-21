import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from stock_predictor import load_and_preprocess_data, build_lstm_model, train_and_predict, predict_future
import streamlit as st

def main():
    import streamlit as st
    st.set_page_config(
        page_title="Stock Market Trend Prediction",
        page_icon="JGKlogo1.png",
        layout="wide",
    )
    #Main Page Content
    st.title('Stock Market Trend Prediction using LSTM')

    #Information about Page
    st.markdown("""
        Welcome to the **Stock Market Predictor App**!

        This application uses a powerful **Long Short-Term Memory (LSTM)** neural network model to predict stock price trends based on historical data. 
    """)

    with st.expander("📘 Learn How LSTM Predicts Stock Prices"):
        st.markdown("""
            **🔁 What is LSTM?**  
            Long Short-Term Memory (LSTM) is a special kind of **Recurrent Neural Network (RNN)** capable of learning long-term dependencies — perfect for **time-series prediction** like stock prices.

            ---
            **📈 How It Works:**
            LSTM uses a sequence of past values (called time steps) to predict the next value.  
            For example, if you feed it 60 days of stock prices, it tries to predict the 61st day.

            ---
            **🧠 LSTM Cell Key Equations:**
            Each LSTM cell maintains a memory (`C_t`) and decides what to keep, update, or forget using gates:

            - **Forget Gate:**  
            $$
            f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)
            $$

            - **Input Gate:**  
            $$
            i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i) \\\\
            \\tilde{C}_t = \\tanh(W_C \\cdot [h_{t-1}, x_t] + b_C)
            $$

            - **Memory Update:**  
            $$
            C_t = f_t * C_{t-1} + i_t * \\tilde{C}_t
            $$

            - **Output Gate:**  
            $$
            o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o) \\\\
            h_t = o_t * \\tanh(C_t)
            $$

            ---
            **🧮 Loss Function (MSE):**  
            $$
            MSE = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
            $$
            We try to minimize this difference between actual price ($y$) and predicted price ($\\hat{y}$).

            ---
            **💡 Summary:**
            - LSTM can remember important stock patterns from the past.
            - Trained on previous trends, it learns to forecast the future.
            - It's not magic — just **math + data + pattern learning**.
        """)

    st.subheader("Workflow :")
    st.markdown("1.  **Dataset Selection** : Choose to use the default dataset or upload your own CSV file.")
    st.markdown("2.  **Data Preview** : View historical stock data before prediction.")
    st.markdown("3.  **Model Settings** : Choose epochs and batch size, then train the model.")
    st.markdown("4.  **Model Testing - Actual vs Estimated Prices** : Visualize how well the model performed by comparing actual stock prices with estimated values on the test dataset.")
    st.markdown("5.  **Model Evaluation** : Check performance using metrics like MSE and RMSE.")
    st.markdown("6.  **Future Prediction Settings** : Select how many days to forecast.")
    st.markdown("7.  **Predicted Stock Prices** : See the predicted prices for upcoming days.")
    st.markdown("8.  **Print This Page** : To print this page, click the three dots in the top-right corner and select Print")


    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("""
        **Note:** 
        1. Column names must be: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.  
        2. Date must be in the format: `yyyy-mm-dd`.

         ❗**After uploading a file for processing, Please wait at least 5 minutes for the results.**
    """)
    use_default = st.checkbox("Use Default Dataset Instead ✔️")

    # Wait for user action
    if uploaded_file is None and not use_default:
        st.warning("⚠️ Please upload a dataset or **Check ✅** **'Use Default Dataset'** to proceed.")
        st.stop()
    if uploaded_file is not None:
        st.success("✅ Custom dataset uploaded.")
        data_file = uploaded_file
        df = pd.read_csv(uploaded_file)
    else:
        st.success("✅ Using default dataset.")
        data_file = "dataset/default_stock_data.csv"
        df = pd.read_csv("dataset/default_stock_data.csv")

    st.markdown("""---""")
    st.subheader('Preview of Dataset')
    st.write(df.head())
    st.markdown("""---""")

    # Preprocess and train model
    X_train, X_test, y_train, y_test, scaler, scaled_data = load_and_preprocess_data(data_file)
    model = build_lstm_model(X_train.shape[1:])

    col1, col2 = st.columns([1, 3])  # Adjust width ratio
    with col1:
        CountEpochs = st.number_input("Select Number of Epochs", min_value=1, max_value=50, value=1, step=1, label_visibility="visible")
    epochs = CountEpochs

    batch_size = st.selectbox("Select batch size", options=[8, 16, 32, 64, 128, 256, 512, 1024, 2048], index=6)
    predictions = train_and_predict(X_train, X_test, y_train, y_test, model, epochs=epochs, batch_size=batch_size)

    # Inverse scaling of predictions
    #predictions = scaler.inverse_transform(predictions)

    # Plotting the results
    #st.markdown("""---""")
    #st.subheader('Stock Price Prediction')
    #st.line_chart(predictions)
    st.markdown("""---""")
    st.subheader('Actual vs Estimated Stock Prices')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual Price')
    plt.plot(np.arange(len(predictions)), scaler.inverse_transform(predictions.reshape(-1, 1)), color='red', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Actual vs Estimated Stock Prices')
    plt.legend()
    st.pyplot(plt)

    # Inverse transform predicted and actual values
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)

    # Display metrics in Streamlit
    st.markdown("""---""")
    st.markdown("### Model Performance Metrics")
    st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
    st.write(f"**MSE (Mean Squared Error):** {mse:.4f}")
    st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    #st.write(f"**The model is highly reliable, accurately predicting stock prices with {r2:.2f%} accuracy. ** ")
    st.markdown(f"**The model is highly reliable, explaining {r2:.2%} of the variability in stock prices.**")



    # -----------------------------------
    # 🔮 FUTURE PREDICTION GRAPH SECTION
    # Future Prediction: e.g., next 7 days
    # -----------------------------------
    st.markdown("""---""")
    st.write("### Predict Stock Performance in the Coming Days")

    col3, col4 = st.columns([1, 3])  # Adjust width ratio
    with col3:
        NoOfDatesToPredict = st.number_input("Days", min_value=1, max_value=30, value=7, step=1, label_visibility="collapsed")

    st.write(f"You selected: {NoOfDatesToPredict} day(s)")
    n_days = NoOfDatesToPredict
    last_sequence = scaled_data[-60:].reshape(-1)
    future_preds = predict_future(model, last_sequence, n_days=n_days, scaler=scaler)

    # Create day labels (Day 1, Day 2, ..., Day n)
    day_labels = [f"Day {i+1}" for i in range(n_days)]

    # Plot future predictions
    st.subheader(f"📅 Next {n_days} Days Stock Price Prediction")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(day_labels, future_preds, marker='o', linestyle='-', color='green')
    ax.set_xlabel("Future Days")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Stock Price Forecast for Next {n_days} Days")
    ax.grid(True)

    st.pyplot(fig)



    
    #Copyright Mark
    st.markdown(""" --- 
         © Copyright 2025 J.Gopala Krishna, All rights reserved | Built using Streamlit.
    """)

if __name__ == "__main__":
    main()