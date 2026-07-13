# ✈️ FlightPulse AI - Airline Passenger Forecasting

## Overview

FlightPulse AI is a deep learning-based time series forecasting application that predicts future airline passenger demand using an LSTM (Long Short-Term Memory) neural network.

The project includes a modern Streamlit dashboard for data exploration, model evaluation, and future passenger forecasting.

---

## Features

- Historical airline passenger trend analysis
- Data preprocessing and normalization
- Sequence generation for LSTM training
- Deep Learning forecasting using TensorFlow/Keras
- Interactive Streamlit dashboard
- Plotly visualizations
- Future passenger forecasting
- Forecast export as CSV
- Model evaluation metrics

---

## Dashboard Preview

### Modules Included

- Historical Trend Analysis
- Forecast Generation
- Model Evaluation
- Interactive Visualizations
- CSV Export Functionality

---

## Project Structure

```text
AIRLINE-PASSENGER-FORECAST/
│
├── assets/
│   └── 201623.png
│
├── data/
│   └── airline-passengers.csv
│
├── models/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── outputs/
│   ├── loss_curve.png
│   ├── prediction_plot.png
│   └── forecast_plot.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── sequence_generator.py
│   ├── train.py
│   ├── forecast.py
│   ├── evaluate.py
│   ├── visualization.py
│   ├── predict.py
│   └── model.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the classic Airline Passenger dataset containing monthly passenger counts from 1949 to 1960.

Dataset columns:

- month
- total_passengers

---

## Machine Learning Pipeline

### 1. Data Loading
- Load CSV dataset
- Validate missing values
- Convert date column to datetime format

### 2. Data Preprocessing
- Min-Max Scaling
- Normalization to range [0,1]

### 3. Sequence Generation
- Sliding Window Approach
- Sequence Length = 12 months

### 4. Model Training
- LSTM Neural Network
- Adam Optimizer
- Mean Squared Error Loss

### 5. Forecasting
- Multi-step future prediction
- Recursive forecasting strategy

---

## Model Architecture

```text
Input Layer
     ↓
LSTM Layer (50 Units)
     ↓
Dense Layer
     ↓
Passenger Forecast
```

---

## Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- TensorFlow
- Keras

### Data Processing
- Pandas
- NumPy
- Scikit-learn

### Visualization
- Plotly
- Matplotlib

### Deployment
- Streamlit

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd AIRLINE-PASSENGER-FORECAST
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Forecast Example

Input:
- Previous 12 months passenger data

Output:
- Forecast for future months
- Interactive forecast visualizations
- Downloadable CSV report

---

## Future Improvements

- GRU Model Support
- Transformer-Based Forecasting
- Confidence Intervals
- Hyperparameter Optimization
- Cloud Deployment
- Real-time Data Integration

---

## Author

**Akash B**

B.Tech Data Science  
Anurag University

---

## License

This project is intended for educational and research purposes.