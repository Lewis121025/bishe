import os
import pandas as pd
import numpy as np

def update_symbol_format(input_file, output_file):
    print("Updating Symbol format to 6-digit string...")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # Use converters to ensure Symbol is read as string, preserving leading zeros if present
    # But since current file has them as numbers (1, 2...), we read as is and format
    df = pd.read_csv(input_file)
    
    # Check if Symbol is numeric or string
    # If numeric (1, 2), format to '000001'
    # If string ('1', '2'), format to '000001'
    
    # 1. Convert to numeric first to handle any mixed types safely (coercing errors)
    # Then format to 6-digit string
    df['Symbol'] = pd.to_numeric(df['Symbol'], errors='coerce').fillna(0).astype(int)
    df['Symbol'] = df['Symbol'].apply(lambda x: f"{x:06d}")
    
    # Sort for tidiness
    df.sort_values(by=['Symbol', 'Year'], inplace=True)
    
    print(f"Sample Symbols: {df['Symbol'].head(3).tolist()}")
    
    df.to_csv(output_file, index=False)
    print(f"Saved updated dataset to {output_file}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    target_file = os.path.join(current_dir, "final_dataset.csv")
    
    update_symbol_format(target_file, target_file)
