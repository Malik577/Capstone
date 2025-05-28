import pandas as pd

def convert_excel_to_csv():
    # Read the Excel file
    excel_file = 'IPO_data_to_learn.xlsx'
    df = pd.read_excel(excel_file)
    
    # Create the CSV filename by replacing .xlsx with .csv
    csv_file = excel_file.replace('.xlsx', '.csv')
    
    # Convert to CSV
    df.to_csv(csv_file, index=False)
    print(f"Successfully converted {excel_file} to {csv_file}")

if __name__ == "__main__":
    convert_excel_to_csv()
