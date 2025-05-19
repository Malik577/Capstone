#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
from joblib import dump
import sys

# Add parent directory to path to enable relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.clean_data import clean_data
from preprocessing.encode_categorical import encode_categorical_features
from preprocessing.impute_missing import impute_numeric_features
from preprocessing.feature_engineering import apply_feature_engineering, select_features
from models.xgboost_model import train_xgboost_model, save_xgboost_model
from models.random_forest_model import train_random_forest_model, save_random_forest_model
from models.gradient_boost_model import train_gradient_boost_model, save_gradient_boost_model
from models.ensemble_model import train_ensemble_model, save_ensemble_model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train machine learning models for IPO price prediction')
    
    parser.add_argument('--input-path', type=str, default='data/raw/training_data.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output-path', type=str, default='models/trained',
                        help='Directory to save trained models')
    parser.add_argument('--model', type=str, default='ensemble',
                        choices=['xgboost', 'random_forest', 'gradient_boost', 'ensemble', 'all'],
                        help='Model type to train')
    parser.add_argument('--target', type=str, default='offerPrice',
                        choices=['offerPrice', 'closeDay1', 'both'],
                        help='Target variable to predict')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size as a fraction of the data')
    parser.add_argument('--use-robust-scaler', action='store_true',
                        help='Use RobustScaler instead of StandardScaler')
    parser.add_argument('--select-features', action='store_true',
                        help='Use feature selection')
    parser.add_argument('--apply-feature-engineering', action='store_true',
                        help='Apply feature engineering')
    
    return parser.parse_args()

def calculate_metrics(y_true, y_pred):
    """Calculate MSE, RMSE, and R2 metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

def main():
    """Main function to execute the training process."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_path}")
    try:
        data = pd.read_csv(args.input_path)
    except FileNotFoundError:
        print(f"Error: File {args.input_path} not found")
        return
    
    # Print column names to help with debugging
    print("Available columns in dataset:")
    print(list(data.columns))
    
    print("\nDataset information:")
    print(f"Shape: {data.shape}")
    print("\nSample data (first 2 rows):")
    print(data.head(2))
    
    # We'll use a safer approach: filter columns that actually exist
    # Initialize empty list for numeric features
    numeric_features = []
    
    # Check if columns exist and add them to the list
    for col in ['egc', 'highTech', 'age', 'year', 'exchange', 'industryFF12', 'nUnderwriters',
               'sharesOfferedPerc', 'investmentReceived', 'amountOnProspectus',
               'commonEquity', 'sp2weeksBefore', 'blueSky', 'managementFee',
               'bookValue', 'totalAssets', 'totalRevenue', 'netIncome',
               'roa', 'leverage', 'vc', 'pe', 'prominence', 'nVCs', 'nExecutives',
               'priorFinancing', 'reputationLeadMax', 'reputationAvg', 'nPatents']:
        if col in data.columns:
            numeric_features.append(col)
    
    # Handle ipoSize as a special case
    if 'ipoSize' in data.columns:
        # Access it directly to make sure it works
        try:
            test_val = data['ipoSize'].iloc[0]
            print(f"ipoSize test value: {test_val}")
            numeric_features.append('ipoSize')
        except Exception as e:
            print(f"Error accessing ipoSize: {e}")
            print("ipoSize will not be included in features")
    
    # Check for ipoSize_normalized
    if 'ipoSize_normalized' in data.columns:
        print("Found ipoSize_normalized in dataset")
    
    print("\nSelected numeric features:", numeric_features)
    print(f"Total: {len(numeric_features)} features")
    
    # Preprocess data
    print("\nPreprocessing data...")
    data = clean_data(data)
    data = encode_categorical_features(data)
    
    # Define targets
    targets = []
    if args.target in ['offerPrice', 'both']:
        targets.append('offerPrice')
    if args.target in ['closeDay1', 'both']:
        targets.append('closeDay1')
    
    print(f"\nTarget variables: {targets}")
    
    # Dictionary to store predictions for second-stage model
    offer_predictions = None
    
    for target_idx, target in enumerate(targets):
        print(f"\nTraining models for target: {target}")
        
        # Prepare features and target
        try:
            # We'll create X more carefully
            X = pd.DataFrame()
            for col in numeric_features:
                if col in data.columns:
                    X[col] = data[col].copy()
                else:
                    print(f"Warning: Column {col} not found in data")
            
            if 'ipoSize_normalized' in data.columns:
                X['ipoSize_normalized'] = data['ipoSize_normalized']
            
            # Add predicted offer price for closeDay1 prediction
            if target == 'closeDay1' and target_idx > 0 and offer_predictions is not None:
                X['predicted_offerPrice'] = offer_predictions
                
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
        
        # Drop rows with NaN in target
        if target in data.columns:
            mask = ~data[target].isna()
            X_filtered = X[mask]
            y_filtered = data.loc[mask, target]
        else:
            print(f"Error: Target '{target}' not found in data")
            continue
        
        print(f"After filtering, X has {X_filtered.shape[0]} rows and {X_filtered.shape[1]} columns")
        print(f"y has {len(y_filtered)} values")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=args.test_size, random_state=42
        )
        
        # Impute missing values
        print("Imputing missing values...")
        X_train_imputed, imputer = impute_numeric_features(X_train)
        X_test_imputed, _ = impute_numeric_features(X_test, imputer)
        
        # Feature selection if requested
        if args.select_features:
            print("Performing feature selection...")
            try:
                feature_selector = select_features(X_train_imputed, y_train)
                X_train_selected = feature_selector.transform(X_train_imputed)
                X_test_selected = feature_selector.transform(X_test_imputed)
                dump(feature_selector, os.path.join(args.output_path, f'feature_selector_{target}.joblib'))
                print(f"After feature selection, X_train has shape {X_train_selected.shape}")
            except Exception as e:
                print(f"Error during feature selection: {e}")
                print("Continuing without feature selection")
                X_train_selected = X_train_imputed
                X_test_selected = X_test_imputed
        else:
            X_train_selected = X_train_imputed
            X_test_selected = X_test_imputed
        
        # Scale features
        print("Scaling features...")
        if args.use_robust_scaler:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        print(f"After scaling, X_train has shape {X_train_scaled.shape}")
        
        # Save preprocessors
        dump(imputer, os.path.join(args.output_path, f'imputer_{target}.joblib'))
        dump(scaler, os.path.join(args.output_path, f'scaler_{target}.joblib'))
        
        # Train models based on specified model type
        models_to_train = []
        if args.model in ['xgboost', 'all']:
            models_to_train.append('xgboost')
        if args.model in ['random_forest', 'all']:
            models_to_train.append('random_forest')
        if args.model in ['gradient_boost', 'all']:
            models_to_train.append('gradient_boost')
        if args.model in ['ensemble', 'all']:
            models_to_train.append('ensemble')
        
        # Dictionary to store trained models
        trained_models = {}
        model_predictions = {}
        
        for model_type in models_to_train:
            print(f"Training {model_type} model for {target}")
            
            if model_type == 'xgboost':
                model = train_xgboost_model(X_train_scaled, y_train)
                save_xgboost_model(model, os.path.join(args.output_path, f'xgboost_{target}.joblib'))
                trained_models['xgboost'] = model
                
                # Evaluate the model
                predictions = model.predict(X_test_scaled)
                model_predictions['xgboost'] = predictions
                mse, rmse, r2 = calculate_metrics(y_test, predictions)
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
            elif model_type == 'random_forest':
                model = train_random_forest_model(X_train_scaled, y_train)
                save_random_forest_model(model, os.path.join(args.output_path, f'random_forest_{target}.joblib'))
                trained_models['random_forest'] = model
                
                # Evaluate the model
                predictions = model.predict(X_test_scaled)
                model_predictions['random_forest'] = predictions
                mse, rmse, r2 = calculate_metrics(y_test, predictions)
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
            elif model_type == 'gradient_boost':
                model = train_gradient_boost_model(X_train_scaled, y_train)
                save_gradient_boost_model(model, os.path.join(args.output_path, f'gradient_boost_{target}.joblib'))
                trained_models['gradient_boost'] = model
                
                # Evaluate the model
                predictions = model.predict(X_test_scaled)
                model_predictions['gradient_boost'] = predictions
                mse, rmse, r2 = calculate_metrics(y_test, predictions)
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
                
            elif model_type == 'ensemble':
                # If we're only training the ensemble, we need to train the base models first
                if 'xgboost' not in trained_models:
                    xgb_model = train_xgboost_model(X_train_scaled, y_train)
                    trained_models['xgboost'] = xgb_model
                    model_predictions['xgboost'] = xgb_model.predict(X_test_scaled)
                
                if 'random_forest' not in trained_models:
                    rf_model = train_random_forest_model(X_train_scaled, y_train)
                    trained_models['random_forest'] = rf_model
                    model_predictions['random_forest'] = rf_model.predict(X_test_scaled)
                
                if 'gradient_boost' not in trained_models:
                    gb_model = train_gradient_boost_model(X_train_scaled, y_train)
                    trained_models['gradient_boost'] = gb_model
                    model_predictions['gradient_boost'] = gb_model.predict(X_test_scaled)
                
                # Train the ensemble model
                model = train_ensemble_model(X_train_scaled, y_train)
                save_ensemble_model(model, os.path.join(args.output_path, f'ensemble_{target}.joblib'))
                
                # Evaluate the model
                predictions = model.predict(X_test_scaled)
                mse, rmse, r2 = calculate_metrics(y_test, predictions)
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        # Save all models in a dictionary if 'all' or 'ensemble' is specified
        if args.model in ['all', 'ensemble']:
            dump(trained_models, os.path.join(args.output_path, f'{target}_models.joblib'))
        
        # Create average prediction for all trained models if we need it for closeDay1
        if target == 'offerPrice' and 'closeDay1' in targets:
            # Make predictions on the full dataset for closeDay1 model
            try:
                X_full = pd.DataFrame()
                for col in numeric_features:
                    if col in data.columns:
                        X_full[col] = data[col].copy()
                
                if 'ipoSize_normalized' in data.columns:
                    X_full['ipoSize_normalized'] = data['ipoSize_normalized']
                
                if args.apply_feature_engineering:
                    try:
                        X_full = apply_feature_engineering(X_full)
                    except Exception as e:
                        print(f"Error during feature engineering for full dataset: {e}")
                    
                X_full_imputed, _ = impute_numeric_features(X_full, imputer)
                
                if args.select_features and 'feature_selector' in locals():
                    try:
                        X_full_selected = feature_selector.transform(X_full_imputed)
                    except Exception as e:
                        print(f"Error during feature selection for full dataset: {e}")
                        X_full_selected = X_full_imputed
                else:
                    X_full_selected = X_full_imputed
                    
                X_full_scaled = scaler.transform(X_full_selected)
                
                # Generate predictions from all trained models
                full_predictions = np.zeros(len(X_full_scaled))
                for model in trained_models.values():
                    full_predictions += model.predict(X_full_scaled)
                full_predictions /= len(trained_models)
                
                # Store predictions for use in closeDay1 model
                offer_predictions = full_predictions
            except Exception as e:
                print(f"Error making predictions for closeDay1: {e}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
