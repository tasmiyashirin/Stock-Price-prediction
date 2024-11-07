# Stock-Price-prediction
This project uses machine learning techniques to predict stock prices based on historical data. The model leverages LSTM (Long Short-Term Memory) networks for time series forecasting. The project is built using TensorFlow, Keras, and Streamlit for deployment.

## Features

- **Data Processing**: The historical stock prices are fetched and preprocessed to suit the input format for the model.
- **Model Architecture**: The model is based on LSTM layers for predicting future stock prices.
- **Evaluation**: The model's performance is evaluated using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.
- **Streamlit Deployment**: The app is deployed using Streamlit for an interactive, user-friendly interface.

## Installation

#### 1. Clone the repository:

```
git clone https://github.com/tasmiyashirin/Stock-Price-prediction.git
```

#### 2. Install dependencies: Create a virtual environment (optional, but recommended):

```
python -m venv venv
```

##### Activate the virtual environment:

- For Windows:
```
venv\Scripts\activate
```
- For macOS/Linux:
```
source venv/bin/activate
```

##### Install the required libraries:

```
pip install -r requirements.txt
```

#### 3. Run the application:

- To run the Streamlit app, use the following command:
```
streamlit run app.py
```

#### 4. The application should open in your default browser.


## Project Structure
```
Stock-Price-prediction/
├── app.py              # Streamlit application file
├── keras_model.keras   # Trained LSTM model
├── SPP.ipynb           # Jupyter notebook for initial analysis and training
├── requirements.txt    # List of required packages
└── README.md           # This file
```

## Model Evaluation
- **Mean Absolute Error (MAE):** 1.92
- **Root Mean Squared Error (RMSE):** 2.33
- **R² Score:** 0.93

## Acknowledgments
- Thanks to TensorFlow for providing the machine learning framework.
- Streamlit for making deployment easier.

