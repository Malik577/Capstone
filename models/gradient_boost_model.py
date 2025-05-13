import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

def create_gradient_boost_model(params=None):
    """
    Create a Gradient Boosting regression model with specified parameters
    
    Parameters:
    -----------
    params : dict, optional
        Parameters for GradientBoostingRegressor
        
    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor
        Configured Gradient Boosting model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'random_state': 42
        }
    
    return GradientBoostingRegressor(**params)

def train_gradient_boost_model(X_train, y_train, params=None):
    """
    Train a Gradient Boosting regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Target values
    params : dict, optional
        Parameters for GradientBoostingRegressor
        
    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    """
    model = create_gradient_boost_model(params)
    model.fit(X_train, y_train)
    return model

def predict_gradient_boost(model, X):
    """
    Make predictions using a Gradient Boosting model
    
    Parameters:
    -----------
    model : sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    X : pandas.DataFrame
        Features to predict on
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    """
    return model.predict(X)

def save_gradient_boost_model(model, filename):
    """
    Save Gradient Boosting model to file
    
    Parameters:
    -----------
    model : sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    filename : str
        Path to save the model
    """
    dump(model, filename)

def load_gradient_boost_model(filename):
    """
    Load Gradient Boosting model from file
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor
        Loaded Gradient Boosting model
    """
    return load(filename)
