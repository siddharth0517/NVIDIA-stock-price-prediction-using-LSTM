# NVIDIA Stock Price Prediction Using LSTM

## Overview

This project predicts NVIDIA's stock price using a **Long Short-Term Memory** (LSTM) model. It involves **data collection, exploratory data analysis (EDA), preprocessing, model training, evaluation, and deployment** using Streamlit.

![image](https://github.com/user-attachments/assets/349686d3-d24a-49ff-9f90-0c09e99a286d)


## Features

- **Stock Data Collection**: Uses `yfinance` to fetch historical NVIDIA stock data.
- **Exploratory Data Analysis**: Visualizations for outlier detection, moving averages, and feature correlations.
- **Data Preprocessing**: Normalization using `MinMaxScaler` and sequence creation for LSTM.
- **LSTM Model**: Built using TensorFlow and Keras with dropout layers to prevent overfitting.
- **Model Evaluation**: Uses RMSE and visualization for accuracy assessment.
- **Future Price Prediction**: Predicts the next day's stock price.
- **Deployment**: Implemented using Streamlit for an interactive web app.

## Technologies Used

- Python
- TensorFlow/Keras
- Scikit-learn
- Matplotlib & Seaborn
- Pandas & NumPy
- yFinance
- Streamlit

## Project Structure

```
|-- app.py                 # Streamlit application file
|-- NVIDIA_LSTM.ipynb      # Jupyter Notebook for model training
|-- nvda_lstm_model.keras  # Trained LSTM model
|-- scaler_train.npy       # Scaler object for training data
|-- scaler_test.npy        # Scaler object for test data
|-- README.md              # Project documentation
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/siddharth0517/nvda-stock-prediction.git
   cd nvda-stock-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## Usage

- The web app allows users to visualize historical stock data and predicted prices.
- The model provides a forecast for the next day’s NVIDIA stock price.

![image](https://github.com/user-attachments/assets/254e60ab-5ec3-4d63-ade0-f83d8c9b3048)



## Results

- The model's performance is evaluated using RMSE.
- The RMSE score achieved by the model is **5.84**.
- A comparison of actual vs. predicted stock prices is plotted.

## Future Improvements

- Enhancing model accuracy with more features (e.g., news sentiment, macroeconomic indicators).
- Implementing hyperparameter tuning.
- Extending the model to predict prices for multiple stocks.

## Author

- **Siddharth Jaiswal**
- Contact: siddharthjaiswalvns123@gmail.com
- GitHub: [siddharth0517](https://github.com/siddharth0517)

## License

This project is open-source and available under the [MIT License](LICENSE).

