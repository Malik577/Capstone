import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def create_imputer(strategy='mean'):
    """
    Create a SimpleImputer with the specified strategy
    
    Parameters:
    -----------
    strategy : str, default='mean'
        The imputation strategy. One of 'mean', 'median', 'most_frequent', 'constant'
        
    Returns:
    --------
    sklearn.impute.SimpleImputer
        Configured imputer
    """
    return SimpleImputer(strategy=strategy)

def fit_imputer(imputer, data):
    """
    Fit an imputer on data
    
    Parameters:
    -----------
    imputer : sklearn.impute.SimpleImputer
        Imputer to fit
    data : pandas.DataFrame
        Data to fit the imputer on
        
    Returns:
    --------
    sklearn.impute.SimpleImputer
        Fitted imputer
    """
    imputer.fit(data)
    return imputer

def impute_numeric_features(data, imputer=None, strategy='mean'):
    """
    Impute missing values in numeric features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
    imputer : sklearn.impute.SimpleImputer, optional
        Pre-fitted imputer. If None, a new one will be created and fitted
    strategy : str, default='mean'
        Imputation strategy if creating a new imputer
        
    Returns:
    --------
    tuple
        (imputed_data: pandas.DataFrame, imputer: sklearn.impute.SimpleImputer)
    """
    if imputer is None:
        imputer = create_imputer(strategy=strategy)
        imputer = fit_imputer(imputer, data)
    
    # Get column names to restore them after imputation
    columns = data.columns
    
    # Perform imputation
    imputed_data = imputer.transform(data)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=columns)
    
    return imputed_df, imputer
