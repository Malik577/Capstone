#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from joblib import load
import sys
import math
from sklearn.metrics import mean_squared_error, r2_score

# Add parent directory to path to enable relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_data import clean_data
from preprocessing.encode_categorical import encode_categorical_features
from preprocessing.impute_missing import impute_numeric_features
from preprocessing.feature_engineering import apply_feature_engineering

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    
    parser.add_argument('--input-path', type=str, default='data/raw/testing.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output-path', type=str, default='data/predictions/predictions.csv',
                        help='Path to save the output CSV file with predictions')
    parser.add_argument('--model-path', type=str, default='models/trained',
                        help='Directory containing trained models and preprocessors')
    parser.add_argument('--model-type', type=str, default='ensemble',
                        choices=['xgboost', 'random_forest', 'gradient_boost', 'ensemble', 'all'],
                        help='Type of model to use for prediction')
    parser.add_argument('--target', type=str, default='both',
                        choices=['offerPrice', 'closeDay1', 'both'],
                        help='Target variable to predict')
    parser.add_argument('--apply-feature-engineering', action='store_true',
                        help='Apply feature engineering')
    parser.add_argument('--select-features', action='store_true',
                        help='Use feature selection')
    
    return parser.parse_args()

def load_model(model_path, model_type, target):
    """
    Load the specified model for prediction
    
    Parameters:
    -----------
    model_path : str
        Path to the directory containing models
    model_type : str
        Type of model to load
    target : str
        Target variable
        
    Returns:
    --------
    object
        Loaded model
    """
    if model_type == 'all':
        return load(os.path.join(model_path, f'{target}_models.joblib'))
    else:
        return load(os.path.join(model_path, f'{model_type}_{target}.joblib'))

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics if actual values are available."""
    if y_true is None or len(y_true) == 0:
        return None, None, None
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def main():
    """Main function to execute the prediction process."""
    args = parse_arguments()
    
    # Check if models directory exists
    if not os.path.exists(args.model_path):
        print(f"Error: Models directory '{args.model_path}' does not exist")
        print("Please ensure you have trained models in the specified directory")
        return
    
    # Load data
    print(f"Loading data from {args.input_path}")
    try:
        data = pd.read_csv(args.input_path)
    except FileNotFoundError:
        print(f"Error: File {args.input_path} not found")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Define potential features (all possible columns we might use)
    potential_features = ['egc', 'highTech', 'age', 'year', 'exchange', 'industryFF12', 'nUnderwriters',
                      'sharesOfferedPerc', 'investmentReceived', 'amountOnProspectus',
                      'commonEquity', 'sp2weeksBefore', 'blueSky', 'managementFee',
                      'bookValue', 'totalAssets', 'totalRevenue', 'netIncome',
                      'roa', 'leverage', 'vc', 'pe', 'prominence', 'nVCs', 'nExecutives',
                      'priorFinancing', 'reputationLeadMax', 'reputationAvg', 'nPatents',
                      'ipoSize', 'ipoSize_normalized']

    # Add encoded columns if present in the data
    encoded_columns = [col for col in data.columns if col.startswith('exchange_encoded') or col.startswith('industry_')]
    all_features = potential_features + encoded_columns

    # Add any missing columns with default value (NaN)
    for col in all_features:
        if col not in data.columns:
            data[col] = np.nan

    # Use all features that are now present
    numeric_features = [col for col in all_features if col in data.columns]

    print(f"Using {len(numeric_features)} numeric features: {numeric_features}")
    
    # Preprocess data
    print("Preprocessing data...")
    data = clean_data(data)
    data = encode_categorical_features(data)
    
    # Define targets
    targets = []
    if args.target in ['offerPrice', 'both']:
        targets.append('offerPrice')
    if args.target in ['closeDay1', 'both']:
        targets.append('closeDay1')
    
    for target_idx, target in enumerate(targets):
        print(f"\nMaking predictions for target: {target}")
        
        # Prepare features
        try:
            # Create feature matrix
            X = pd.DataFrame()
            for col in numeric_features:
                if col in data.columns:
                    X[col] = data[col].copy()
            
            # Add predicted offer price for closeDay1 prediction
            if target == 'closeDay1' and 'predicted_offerPrice' in data.columns:
                X['predicted_offerPrice'] = data['predicted_offerPrice']
                
            print(f"Feature matrix X shape: {X.shape}")
        except Exception as e:
            print(f"Error creating feature matrix: {e}")
            continue
        
        # Apply feature engineering if requested
        if args.apply_feature_engineering:
            print("Applying feature engineering...")
            try:
                X = apply_feature_engineering(X)
                print(f"After feature engineering, X has shape {X.shape}")
            except Exception as e:
                print(f"Error during feature engineering: {e}")
                print("Continuing without feature engineering")
        
        # Load preprocessors
        try:
            imputer_path = os.path.join(args.model_path, f'imputer_{target}.joblib')
            scaler_path = os.path.join(args.model_path, f'scaler_{target}.joblib')
            
            if not os.path.exists(imputer_path):
                print(f"Error: Imputer file not found at {imputer_path}")
                print("Please ensure you have trained the models first")
                continue
                
            if not os.path.exists(scaler_path):
                print(f"Error: Scaler file not found at {scaler_path}")
                print("Please ensure you have trained the models first")
                continue
                
            imputer = load(imputer_path)
            scaler = load(scaler_path)

            # Subset X to only the columns the imputer was trained on
            if hasattr(imputer, 'feature_names_in_'):
                expected_features = list(imputer.feature_names_in_)
                missing_features = [col for col in expected_features if col not in X.columns]
                extra_features = [col for col in X.columns if col not in expected_features]
                if missing_features:
                    print(f"Warning: Missing features in test set: {missing_features}. Filling with NaN.")
                    for col in missing_features:
                        X[col] = np.nan
                if extra_features:
                    print(f"Note: Extra features in test set not used by model: {extra_features}. They will be ignored.")
                X = X[expected_features]
        except Exception as e:
            print(f"Error loading preprocessors: {e}")
            print(f"Make sure the model files exist in {args.model_path}")
            continue
        
        # Apply preprocessing
        X_imputed, _ = impute_numeric_features(X, imputer)
        
        # Apply feature selection if requested
        if args.select_features:
            try:
                feature_selector = load(os.path.join(args.model_path, f'feature_selector_{target}.joblib'))
                X_selected = feature_selector.transform(X_imputed)
                print(f"After feature selection, X has shape {X_selected.shape}")
            except FileNotFoundError:
                print(f"Warning: Feature selector for {target} not found. Skipping feature selection.")
                X_selected = X_imputed
        else:
            X_selected = X_imputed
        
        # Apply scaling
        try:
            X_scaled = scaler.transform(X_selected)
            print(f"After scaling, X has shape {X_scaled.shape}")
        except ValueError as e:
            print(f"Error during scaling: {e}")
            print("This is likely due to mismatched columns between training and testing data.")
            
            # Try to determine the expected features from the training data
            try:
                # If we can't directly access the feature names from the scaler,
                # let's create a dummy feature matrix with matching dimensions
                feature_count = scaler.n_features_in_
                print(f"Expected feature count: {feature_count}")
                
                if len(X_selected.shape) == 1:
                    # If X_selected is 1D, reshape it
                    X_selected = X_selected.reshape(-1, 1)
                
                # If dimensions don't match, we need to adjust
                if X_selected.shape[1] != feature_count:
                    print(f"Feature mismatch: have {X_selected.shape[1]}, need {feature_count}")
                    # Create a properly sized array of zeros
                    X_scaled = np.zeros((X_selected.shape[0], feature_count))
                    # Copy over available features (up to the min size)
                    min_cols = min(X_selected.shape[1], feature_count)
                    X_scaled[:, :min_cols] = X_selected[:, :min_cols]
                else:
                    X_scaled = scaler.transform(X_selected)
            except Exception as inner_e:
                print(f"Error fixing feature mismatch: {inner_e}")
                print("Unable to make predictions for this target.")
                continue
        
        # Load model and make predictions
        try:
            if args.model_type == 'all':
                # When using all models, average the predictions
                models = load_model(args.model_path, args.model_type, target)
                
                predictions = np.zeros(len(X_scaled))
                for model_name, model in models.items():
                    model_predictions = model.predict(X_scaled)
                    predictions += model_predictions
                    print(f"  {model_name} predictions: mean={model_predictions.mean():.4f}, std={model_predictions.std():.4f}")
                predictions /= len(models)
            else:
                # Otherwise use the specified model
                model = load_model(args.model_path, args.model_type, target)
                predictions = model.predict(X_scaled)
            
            # Add predictions to the dataset
            data[f'predicted_{target}'] = predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            continue
        
        # Evaluate predictions if actual values are available
        if target in data.columns:
            actual_values = data[target].dropna()
            pred_values = data.loc[~data[target].isna(), f'predicted_{target}']
            if len(actual_values) > 0:
                mse, rmse, r2 = calculate_metrics(actual_values, pred_values)
                print(f"\nEvaluation metrics for {target}:")
                print(f"  MSE: {mse:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R2: {r2:.4f}")
    
    # Save the results
    print(f"Saving predictions to {args.output_path}")
    prediction_cols = [f'predicted_{target}' for target in targets if f'predicted_{target}' in data.columns]
    if prediction_cols:
        data[prediction_cols].to_csv(args.output_path, index=False)
    else:
        data.to_csv(args.output_path, index=False)
    
    # Display a sample of predictions
    prediction_cols = [f'predicted_{target}' for target in targets if f'predicted_{target}' in data.columns]
    if prediction_cols:
        print("\nSample of predictions:")
        print(data[prediction_cols].head())
        print("\nPrediction completed successfully!")
    else:
        print("\nNo predictions were generated. Check the errors above.")

if __name__ == "__main__":
    main()
