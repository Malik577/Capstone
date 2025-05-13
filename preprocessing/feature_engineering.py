import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

def create_interaction_features(X):
    """
    Create interaction features between existing features
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input feature DataFrame
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added interaction features
    """
    X_copy = X.copy()
    
    # Create interaction features
    X_copy['revenue_per_asset'] = X_copy['totalRevenue'] / (X_copy['totalAssets'] + 1e-6)
    X_copy['income_per_revenue'] = X_copy['netIncome'] / (X_copy['totalRevenue'] + 1e-6)
    X_copy['equity_per_asset'] = X_copy['commonEquity'] / (X_copy['totalAssets'] + 1e-6)
    X_copy['investment_per_share'] = X_copy['investmentReceived'] / (X_copy['sharesOfferedPerc'] + 1e-6)
    
    # Create ratio features
    X_copy['vc_to_exec_ratio'] = X_copy['nVCs'] / (X_copy['nExecutives'] + 1e-6)
    X_copy['patent_to_revenue'] = X_copy['nPatents'] / (X_copy['totalRevenue'] + 1e-6)
    
    return X_copy

def select_features(X, y, threshold='median'):
    """
    Select important features using Random Forest feature importance
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target variable
    threshold : str or float, default='median'
        Threshold for feature selection
        
    Returns:
    --------
    sklearn.feature_selection.SelectFromModel
        Fitted feature selector
    """
    selector = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = SelectFromModel(selector, threshold=threshold)
    selector.fit(X, y)
    return selector

def apply_feature_engineering(X):
    """
    Apply all feature engineering operations
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Input features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    X_engineered = create_interaction_features(X)
    return X_engineered
