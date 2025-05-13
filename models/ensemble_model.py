import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, load

from .xgboost_model import create_xgboost_model
from .random_forest_model import create_random_forest_model
from .gradient_boost_model import create_gradient_boost_model

def create_ensemble_model(xgb_params=None, rf_params=None, gb_params=None, final_estimator=None):
    """
    Create a stacking ensemble model using XGBoost, Random Forest, and Gradient Boosting regressors
    
    Parameters:
    -----------
    xgb_params : dict, optional
        Parameters for XGBoost regressor
    rf_params : dict, optional
        Parameters for Random Forest regressor
    gb_params : dict, optional
        Parameters for Gradient Boosting regressor
    final_estimator : estimator object, optional
        The estimator used to combine the base estimators
        
    Returns:
    --------
    sklearn.ensemble.StackingRegressor
        Configured stacking ensemble model
    """
    if final_estimator is None:
        final_estimator = LinearRegression()
    
    estimators = [
        ('xgb', create_xgboost_model(xgb_params)),
        ('rf', create_random_forest_model(rf_params)),
        ('gb', create_gradient_boost_model(gb_params))
    ]
    
    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5
    )

def train_ensemble_model(X_train, y_train, xgb_params=None, rf_params=None, gb_params=None, final_estimator=None):
    """
    Train a stacking ensemble model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Target values
    xgb_params : dict, optional
        Parameters for XGBoost regressor
    rf_params : dict, optional
        Parameters for Random Forest regressor
    gb_params : dict, optional
        Parameters for Gradient Boosting regressor
    final_estimator : estimator object, optional
        The estimator used to combine the base estimators
        
    Returns:
    --------
    sklearn.ensemble.StackingRegressor
        Trained stacking ensemble model
    """
    model = create_ensemble_model(xgb_params, rf_params, gb_params, final_estimator)
    model.fit(X_train, y_train)
    return model

def predict_ensemble(model, X):
    """
    Make predictions using a stacking ensemble model
    
    Parameters:
    -----------
    model : sklearn.ensemble.StackingRegressor
        Trained stacking ensemble model
    X : pandas.DataFrame
        Features to predict on
        
    Returns:
    --------
    numpy.ndarray
        Predicted values
    """
    return model.predict(X)

def save_ensemble_model(model, filename):
    """
    Save stacking ensemble model to file
    
    Parameters:
    -----------
    model : sklearn.ensemble.StackingRegressor
        Trained stacking ensemble model
    filename : str
        Path to save the model
    """
    dump(model, filename)

def load_ensemble_model(filename):
    """
    Load stacking ensemble model from file
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
        
    Returns:
    --------
    sklearn.ensemble.StackingRegressor
        Loaded stacking ensemble model
    """
    return load(filename)
