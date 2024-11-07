import streamlit as st
import pandas as pd
from datetime import datetime
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Function to download stock data using yfinance
def download_data(ticker_symbol, start_date, end_date):
    try:
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        st.success(f"Downloaded data for {ticker_symbol} from Yahoo Finance.")
        return df
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None  # Return None if download fails

# Streamlit App Title
st.title('Stock Price Prediction') 

# User input for ticker symbol
ticker_symbol = st.text_input('Enter Stock Ticker Symbol:', 'AAPL')

# Download data based on user input
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = '2010-01-01'  # Historical data from 2010 for visualization

# Download data using yfinance
df = yf.Ticker(ticker_symbol).history(start=start_date, end=end_date, actions=True)

# Check if data is available
if df.empty:
    st.error("No data found for the ticker symbol. Please check the symbol and try again.")
else:
    # Display stock data from 2010 - 2024
    st.subheader('Data from 2010 - 2024')
    st.write(df)  

# Visualization: Closing Price vs Time chart using Plotly
st.subheader('Closing Price vs Time Chart')

# Create a Plotly figure
fig = go.Figure()

# Add the closing price trace
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))

# Add labels and titles
fig.update_layout(
    title=f'{ticker_symbol} Closing Price Over Time',
    xaxis_title='Date',
    yaxis_title='Price (in USD)',
    template='plotly_dark',
    autosize=True
    )

# Display the figure in Streamlit
st.plotly_chart(fig)

# Visualization: Closing Price vs Time chart with 100-day Moving Average (MA)
st.subheader('Closing Price vs Time Chart with 100MA')

ma100 = df['Close'].rolling(100).mean()

fig_ma100 = go.Figure()
fig_ma100.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
fig_ma100.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100 Day MA', line=dict(color='red')))

fig_ma100.update_layout(
    title=f'{ticker_symbol} Closing Price and 100-Day Moving Average',
    xaxis_title='Date',
    yaxis_title='Price (in USD)',
    template='plotly_dark',
    autosize=True
    )
st.plotly_chart(fig_ma100)

# Visualization: Closing Price vs Time chart with 100-day and 200-day Moving Averages (MA)
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')

ma200 = df['Close'].rolling(200).mean()

fig_ma100_200 = go.Figure()
fig_ma100_200.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
fig_ma100_200.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines', name='100 Day MA', line=dict(color='red')))
fig_ma100_200.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines', name='200 Day MA', line=dict(color='green')))

fig_ma100_200.update_layout(
    title=f'{ticker_symbol} Closing Price, 100-Day and 200-Day Moving Averages',
    xaxis_title='Date',
    yaxis_title='Price (in USD)',
    template='plotly_dark',
    autosize=True
    )
st.plotly_chart(fig_ma100_200)

# Splitting data into training and testing datasets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])  # Training data (70% of total data)
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])  # Testing data (30% of total data)

# Normalizing data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1)) 
data_training_array = scaler.fit_transform(data_training) 

# Load pre-trained model
model = load_model('keras_model.keras')

# Prepare input data for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Preparing test data for prediction
x_test = [] 
y_test = [] 

for i in range(100, input_data.shape[0]): 
    x_test.append(input_data[i-100:i]) 
    y_test.append(input_data[i, 0]) 

x_test, y_test = np.array(x_test), np.array(y_test) 

# Predicting with the model
y_predicted = model.predict(x_test)
scaler = scaler.scale_  # Get the scaler to reverse the scaling

# Scale back the predicted and actual values to original scale
scale_factor = 1/scaler[0] 
y_predicted = y_predicted * scale_factor 
y_test = y_test * scale_factor 

# Flatten the predicted values for visualization
y_predicted = y_predicted.flatten()

# Visualization: Predicted vs Original Stock Prices using Plotly
st.subheader('Predictions vs Original')

# Create a figure for predictions vs original
fig_predictions = go.Figure()

#To show actual dates on the x-axis 
test_dates = df.index[-len(y_test):]

# Plot original price data
fig_predictions.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Original Price', line=dict(color='blue')))
# Plot predicted price data
fig_predictions.add_trace(go.Scatter(x=test_dates, y=y_predicted, mode='lines', name='Predicted Price', line=dict(color='red')))

fig_predictions.update_layout(
    title='Original vs Predicted Stock Prices',
    xaxis_title='Date',
    yaxis_title='Price (in USD)',
    template='plotly_dark',
    legend_title="Price Type",
    autosize=True
)

# Display the plot in Streamlit
st.plotly_chart(fig_predictions)

# Predicting the next day's stock price
st.subheader('Next Day Predicted Price')

# Fetch the last 100 days of closing prices from testing data
past_100_days = data_testing.tail(100)

# Ensure scaler is initialized as a MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # Re-define it here to avoid issues

# Now, apply scaling on past_100_days
input_data = scaler.fit_transform(past_100_days.values.reshape(-1, 1))

# Reshape input to fit the model's expected 3D input (samples, time steps, features)
x_next_day = np.array([input_data])
# Ensure the input has the shape (1, 100, 1)
x_next_day = np.reshape(x_next_day, (x_next_day.shape[0], x_next_day.shape[1], 1))

# Predict the next day
next_day_prediction = model.predict(x_next_day)
next_day_prediction = next_day_prediction[0][0] * scale_factor  # Scale back to original price

st.write(f"The predicted closing price for the next day is: ${next_day_prediction:.2f}")
