import pandas as pd
import numpy as np
from api.main import preprocess_input, load_imputer, load_feature_selector
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

print("=== Step-by-step Pipeline Debug ===")

# Step 1: Preprocessing
df = pd.DataFrame([data])
print(f"1. Input DataFrame shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

processed = preprocess_input(df, 'offerPrice')
print(f"2. After preprocessing shape: {processed.shape}")
print(f"   Columns: {list(processed.columns)}")

# Step 2: Load components and test
imputer = load_imputer('offerPrice')
feature_selector = load_feature_selector('offerPrice')

print(f"3. Imputer expects {len(imputer.feature_names_in_)} features:")
print(f"   {list(imputer.feature_names_in_)}")

print(f"4. Feature selector expects {feature_selector.n_features_in_} features")

# Step 3: Test imputation
print("\n5. Testing imputation...")
try:
    imputed = imputer.transform(processed)
    print(f"   After imputation shape: {imputed.shape}")
    print(f"   SUCCESS: Imputation works")
except Exception as e:
    print(f"   ERROR in imputation: {e}")

# Step 4: Test feature selection
print("\n6. Testing feature selection...")
try:
    if imputed.shape[1] == feature_selector.n_features_in_:
        selected = feature_selector.transform(imputed)
        print(f"   After feature selection shape: {selected.shape}")
        print(f"   SUCCESS: Feature selection works")
    else:
        print(f"   ERROR: Shape mismatch - got {imputed.shape[1]}, expected {feature_selector.n_features_in_}")
except Exception as e:
    print(f"   ERROR in feature selection: {e}")

print("\n=== Analysis ===")
print("The issue appears to be a mismatch between:")
print("- Imputer trained features vs preprocessing output")
print("- Or preprocessing not creating the full engineered feature set") 