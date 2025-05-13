import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from joblib import dump, load

def create_xgboost_model(params=None):
    """
    Create an XGBoost regression model with specified parameters
    
    Parameters:
    -----------
    params : dict, optional
        Parameters for XGBRegressor
        
    Returns:
    --------
    xgboost.XGBRegressor
        Configured XGBoost model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    return XGBRegressor(**params)

def train_xgboost_model(X_train, y_train, params=None):
    """
    Train an XGBoost regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Target values
    params : dict, optional
        Parameters for XGBRegressor
        
    Returns:
    --------
    xgboost.XGBRegressor
        Trained XGBoost model
    """
    model = create_xgboost_model(params)
    model.fit(X_train, y_train)
    return model

def predict_xgboost(model, X):
    """
    Make predictions using an XGBoost model
    
    Parameters:
    -----------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    X : pandas.DataFrame
        Features to predict on
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    """
    return model.predict(X)

def save_xgboost_model(model, filename):
    """
    Save XGBoost model to file
    
    Parameters:
    -----------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    filename : str
        Path to save the model
    """
    dump(model, filename)

def load_xgboost_model(filename):
    """
    Load XGBoost model from file
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    xgboost.XGBRegressor
        Loaded XGBoost model
    """
    return load(filename)
