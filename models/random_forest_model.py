import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

def create_random_forest_model(params=None):
    """
    Create a Random Forest regression model with specified parameters
    
    Parameters:
    -----------
    params : dict, optional
        Parameters for RandomForestRegressor
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Configured Random Forest model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    return RandomForestRegressor(**params)

def train_random_forest_model(X_train, y_train, params=None):
    """
    Train a Random Forest regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Target values
    params : dict, optional
        Parameters for RandomForestRegressor
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    model = create_random_forest_model(params)
    model.fit(X_train, y_train)
    return model

def predict_random_forest(model, X):
    """
    Make predictions using a Random Forest model
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    X : pandas.DataFrame
        Features to predict on
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    """
    return model.predict(X)

def save_random_forest_model(model, filename):
    """
    Save Random Forest model to file
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    filename : str
        Path to save the model
    """
    dump(model, filename)

def load_random_forest_model(filename):
    """
    Load Random Forest model from file
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Loaded Random Forest model
    """
    return load(filename)
