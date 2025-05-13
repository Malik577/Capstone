import numpy as np
import pandas as pd

def normalize_ipo_size(data):
    """
    Apply log transformation to normalize IPO size
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'ipoSize' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'ipoSize_normalized' column
    """
    data_copy = data.copy()
    
    if 'ipoSize' in data_copy.columns:
        data_copy['ipoSize_normalized'] = np.log(data_copy['ipoSize'] + 1)
        data_copy = data_copy.drop(columns=['ipoSize'])
    
    return data_copy

def clean_data(data):
    """
    Apply all data cleaning operations
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data to be cleaned
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned DataFrame
    """
    # For now, just normalize IPO size
    # Add more cleaning steps as needed
    data_cleaned = normalize_ipo_size(data)
    
    return data_cleaned
