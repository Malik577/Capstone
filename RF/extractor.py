import pandas as pd

def read_csv_to_df():
    # Read the CSV file
    csv_file = 'IPO_data_to_learn.csv'
    df = pd.read_csv(csv_file)
    
    # Extract the risk factors column
    risk_factors = df['rf']
    
    # Display information about risk factors
    print("\nRisk Factors Analysis:")
    print(f"Number of entries: {len(risk_factors)}")
    print("\nBasic Statistics:")
    print(risk_factors.describe())
    print("\nFirst few risk factors:")
    print(risk_factors.head())
    
    return df, risk_factors

if __name__ == "__main__":
    df, risk_factors = read_csv_to_df()
