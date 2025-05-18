# IPO Price Prediction

A machine learning project for predicting IPO prices and first-day closing prices using various regression models.

## Project Structure

```
project/
├── data/
│   └── raw/                          # Raw CSV data files
├── models/                           # Model implementation files
│   ├── xgboost_model.py              # XGBoost model
│   ├── random_forest_model.py        # Random Forest model
│   ├── gradient_boost_model.py       # Gradient Boosting model
│   └── ensemble_model.py             # Stacking ensemble model
├── preprocessing/                    # Data preprocessing modules
│   ├── encode_categorical.py         # Categorical feature encoding
│   ├── clean_data.py                 # Data cleaning functions
│   ├── impute_missing.py             # Missing value imputation
│   └── feature_engineering.py        # Feature engineering functions
├── scripts/                          # Training and prediction scripts
│   ├── train.py                      # Model training script
│   └── predict.py                    # Prediction script
├── requirements.txt                  # Project dependencies
└── README.md                         # Project documentation
```

## Installation

1. Clone the repository
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training a Model

Train models using the `train.py` script:

```
python -m project.scripts.train --input-path data/raw/training.csv --output-path models/trained --model ensemble --target both --apply-feature-engineering --use-robust-scaler
```

Arguments:
- `--input-path`: Path to the input CSV file (default: data/raw/training.csv)
- `--output-path`: Directory to save trained models (default: models/trained)
- `--model`: Model type to train (xgboost, random_forest, gradient_boost, ensemble, all)
- `--target`: Target variable to predict (offerPrice, closeDay1, both)
- `--test-size`: Test set size as a fraction (default: 0.2)
- `--poly-degree`: Degree for polynomial feature transformation (default: 2)
- `--use-robust-scaler`: Use RobustScaler instead of StandardScaler
- `--select-features`: Use feature selection
- `--apply-feature-engineering`: Apply feature engineering

### Making Predictions

Make predictions using the `predict.py` script:

```
python -m project.scripts.predict --input-path data/raw/testing.csv --output-path data/predictions.csv --model-path models/trained --model-type ensemble --target both --apply-feature-engineering
```

Arguments:
- `--input-path`: Path to the input CSV file
- `--output-path`: Path to save the output CSV with predictions
- `--model-path`: Directory containing trained models and preprocessors
- `--model-type`: Type of model to use for prediction (xgboost, random_forest, gradient_boost, ensemble, all)
- `--target`: Target variable to predict (offerPrice, closeDay1, both)
- `--apply-feature-engineering`: Apply feature engineering
- `--select-features`: Use feature selection

### API Usage

The project includes a FastAPI implementation for making predictions via HTTP requests. The API provides endpoints for predicting IPO offer prices and first-day closing prices.

#### Starting the API Server

To start the API server:

```bash
python api/main.py
```

The server will start on http://localhost:8001 by default.

#### Available Endpoints

1. **Health Check**
   ```
   GET /health
   ```
   Returns the health status of the API and its dependencies.

2. **API Metadata**
   ```
   GET /metadata
   ```
   Returns information about the API, including expected features and available endpoints.

3. **Offer Price Prediction**
   ```
   POST /predict/offer-price
   ```
   Predicts the IPO offer price for one or more samples.

4. **First-Day Closing Price Prediction**
   ```
   POST /predict/close-day1
   ```
   Predicts the first-day closing price for one or more samples.

5. **Combined Prediction**
   ```
   POST /predict/combined
   ```
   Predicts both offer price and first-day closing price for one or more samples.

#### Example API Request

```python
import requests
import json

# API endpoint
url = "http://localhost:8001/predict/combined"

# Sample IPO data
data = {
    "samples": [{
        "totalRevenue": 1000000,
        "totalAssets": 2000000,
        "netIncome": 500000,
        "commonEquity": 1500000,
        "investmentReceived": 300000,
        "sharesOfferedPerc": 25,
        "nVCs": 2,
        "nExecutives": 5,
        "nPatents": 3,
        "exchange": "NYSE",
        "industryFF12": "Finance",
        "ipoSize": 10000000,
        "nUnderwriters": 3,
        "amountOnProspectus": 10000000,
        "sp2weeksBefore": 100,
        "blueSky": 0.5,
        "managementFee": 0.07,
        "bookValue": 15,
        "roa": 0.25,
        "leverage": 0.3,
        "vc": 1,
        "pe": 20,
        "prominence": 0.8,
        "priorFinancing": 2000000,
        "reputationLeadMax": 0.9,
        "reputationAvg": 0.85
    }]
}

# Make the request
response = requests.post(url, json=data)
predictions = response.json()
print(json.dumps(predictions, indent=2))
```

#### API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Models

This project implements four types of regression models:

1. **XGBoost**: A gradient boosting framework implementation
2. **Random Forest**: An ensemble of decision trees
3. **Gradient Boosting**: A gradient boosting implementation from scikit-learn
4. **Ensemble Model**: A stacking regressor that combines the above models

## Data Processing

The preprocessing pipeline includes:
- Cleaning data (normalize IPO size)
- Encoding categorical features (exchange, industry)
- Imputing missing values
- Feature selection (optional)
- Feature engineering (optional)
  - Creation of interaction features
  - Creation of ratio features
- Feature scaling (Standard or Robust)
- Polynomial feature transformation

## IPO Price Prediction Workflow

For closeDay1 prediction after offerPrice prediction:
1. First, predict the offer price (offerPrice)
2. Use the predicted offer price as an input feature for closeDay1 prediction
3. This two-stage approach improves the accuracy of first-day closing price predictions

## License

This project is licensed under the MIT License.
