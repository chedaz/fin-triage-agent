"""
etl.py - Run this ONCE to convert your Yearly CSVs into Ticker CSVs
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def process_yearly_files(source_dir='./raw_data', output_dir='./data'):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_data = []
    
    # 1. Load all yearly files
    csv_files = list(source_path.glob("*.csv"))
    logging.info(f"Found {len(csv_files)} yearly files.")
    
    for f in csv_files:
        try:
            # Assuming columns like: Date, Ticker, Open, High, Low, Close, Vol
            df = pd.read_csv(f)
            # Normalize column names
            df.columns = df.columns.str.strip().str.title()
            
            # Map your specific CSV headers if they differ
            # Example: if your CSV has 'Code' instead of 'Ticker'
            if 'Code' in df.columns: 
                df.rename(columns={'Code': 'Ticker'}, inplace=True)
                
            all_data.append(df)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")

    if not all_data:
        return

    # 2. Merge and Group
    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'], errors='coerce')
    full_df.dropna(subset=['Date', 'Ticker'], inplace=True)
    
    # 3. Save individual Ticker files
    unique_tickers = full_df['Ticker'].unique()
    logging.info(f"Extracting {len(unique_tickers)} unique tickers...")
    
    for ticker in unique_tickers:
        ticker_df = full_df[full_df['Ticker'] == ticker].sort_values('Date')
        if len(ticker_df) > 50: # Ignore sparse data
            # Clean filename (remove special chars)
            safe_name = "".join([c for c in str(ticker) if c.isalnum()])
            ticker_df.to_csv(output_path / f"{safe_name}.csv", index=False)
            
    logging.info("ETL Complete. Ready for main.py")

if __name__ == "__main__":
    process_yearly_files()
