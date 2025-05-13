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
