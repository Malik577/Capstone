import numpy as np
import pandas as pd

def encode_exchange(data):
    """
    Encode exchange categorical variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'exchange' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded 'exchange' column
    """
    if 'exchange' not in data.columns:
        return data
        
    exchange_map = {
        'AMEX': 0,
        'NASDQ': 1,
        'NYSE': 2
    }
    
    data_copy = data.copy()
    data_copy['exchange'] = data_copy['exchange'].map(exchange_map)
    return data_copy

def encode_industry(data):
    """
    Encode industryFF12 categorical variable
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'industryFF12' column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with encoded 'industryFF12' column
    """
    if 'industryFF12' not in data.columns:
        return data
        
    industry_map = {
        'Business Equipment -- Computers, Software, and Electronic Equipment': 0,
        'Chemicals and Allied Products': 1,
        "Consumer Durables -- Cars, TV's, Furniture, Household Appliances": 2,
        'Consumer NonDurables -- Food, Tobacco, Textiles, Apparel, Leather, Toys': 3,
        'Finance': 4,
        'Healthcare, Medical Equipment, and Drugs': 5,
        'Manufacturing -- Machinery, Trucks, Planes, Off Furn, Paper, Com Printing': 6,
        'Oil, Gas, and Coal Extraction and Products': 7,
        'Other': 8,
        'Telephone and Television Transmission': 9,
        'Utilities': 10,
        'Wholesale, Retail, and Some Services (Laundries, Repair Shops)': 11
    }
    
    data_copy = data.copy()
    data_copy['industryFF12'] = data_copy['industryFF12'].map(industry_map)
    return data_copy

def encode_boolean_columns(data):
    """
    Convert boolean columns to numeric (1/0)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with potential boolean columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with boolean columns converted to 1/0
    """
    data_copy = data.copy()
    
    for col in data_copy.columns:
        if data_copy[col].dtype == object:
            unique_vals = set(data_copy[col].dropna().unique())
            if unique_vals == {'TRUE', 'FALSE'}:
                data_copy[col] = data_copy[col].map({'TRUE': 1, 'FALSE': 0})
            elif unique_vals == {'true', 'false'}:
                data_copy[col] = data_copy[col].map({'true': 1, 'false': 0})
                
    return data_copy

def encode_categorical_features(data):
    """
    Apply all categorical encoding transformations
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all categorical features encoded
    """
    data = encode_exchange(data)
    data = encode_industry(data)
    data = encode_boolean_columns(data)
    
    return data
