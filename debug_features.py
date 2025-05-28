import pandas as pd
import numpy as np
from api.main import preprocess_input
import joblib

# Test data
data = {
    'egc': 1, 'highTech': 1, 'age': 5, 'year': 2024, 
    'exchange': 'NASDQ', 
    'industryFF12': 'Business Equipment -- Computers, Software, and Electronic Equipment',
    'nUnderwriters': 3, 'sharesOfferedPerc': 25.0, 'investmentReceived': 10000000,
    'amountOnProspectus': 200000000, 'commonEquity': 0.75, 'sp2weeksBefore': 4800.5,
    'blueSky': 50000, 'managementFee': 0.07, 'bookValue': 18.75, 'totalAssets': 75000000,
    'totalRevenue': 35000000, 'netIncome': 7000000, 'roa': 0.093, 'leverage': 0.3,
    'vc': 1, 'pe': 0, 'prominence': 1, 'nVCs': 2, 'nExecutives': 8, 'priorFinancing': 2,
    'reputationLeadMax': 8.2, 'reputationAvg': 7.8, 'nPatents': 15, 'ipoSize': 200000000
}

print("=== Original Data ===")
df = pd.DataFrame([data])
print(f"Original shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

print("\n=== After Preprocessing ===")
processed = preprocess_input(df, 'offerPrice')
print(f"Processed shape: {processed.shape}")
print(f"Processed columns: {list(processed.columns)}")

print("\n=== Model Expectations ===")
# Load model components
imputer = joblib.load('models/trained/imputer_offerPrice.joblib')
feature_selector = joblib.load('models/trained/feature_selector_offerPrice.joblib')

print(f"Imputer expects: {len(imputer.feature_names_in_)} features")
print(f"Imputer features: {list(imputer.feature_names_in_)}")

print(f"Feature selector expects: {feature_selector.n_features_in_} features")

print("\n=== Feature Engineering Details ===")
print("Features added during preprocessing:")
print("- ipoSize_normalized (from ipoSize)")
print("- revenue_per_asset")
print("- income_per_revenue") 
print("- equity_per_asset")
print("- investment_per_share")
print("- vc_to_exec_ratio")
print("- patent_to_revenue")

print(f"\nOriginal features: {len(data)} + 6 engineered = {len(data) + 6}")
print(f"After dropping ipoSize: {len(data) + 6 - 1} = {len(data) + 5}")
print(f"Expected total: {len(data) + 5}") 