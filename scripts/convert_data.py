import os
import pandas as pd
import glob

def process_data(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # List .dta files
    dta_files = glob.glob(os.path.join(input_dir, "*.dta"))
    print(f"Found {len(dta_files)} .dta files.")

    processed_files = set()

    for file_path in dta_files:
        filename = os.path.basename(file_path)
        
        # Check if already processed (as part of a pair)
        if filename in processed_files:
            continue

        base_name, ext = os.path.splitext(filename)
        
        # Handle split files logic (based on filename pattern "两个文件上下拼接")
        if "两个文件上下拼接" in filename:
            # Determine the base prefix to find its partner
            # Assuming format: "Name（两个文件上下拼接）-1.dta"
            prefix = filename.split("（")[0]
            
            # Find all parts for this prefix
            parts = [f for f in dta_files if prefix in os.path.basename(f) and "（两个文件上下拼接）" in os.path.basename(f)]
            parts.sort() # Ensure -1 comes before -2
            
            if not parts:
                print(f"Warning: Could not find matching parts for {filename}")
                continue
                
            print(f"Processing split files for {prefix}: {[os.path.basename(p) for p in parts]}")
            
            dfs = []
            for part in parts:
                try:
                    df = pd.read_stata(part)
                    dfs.append(df)
                    processed_files.add(os.path.basename(part))
                except Exception as e:
                    print(f"Error reading {part}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                output_name = f"{prefix}.csv" # Clean name
                output_path = os.path.join(output_dir, output_name)
                combined_df.to_csv(output_path, index=False)
                print(f"Saved combined file to {output_path}")

        else:
            # Normal single file processing
            print(f"Processing single file: {filename}")
            try:
                df = pd.read_stata(file_path)
                output_name = base_name + ".csv"
                output_path = os.path.join(output_dir, output_name)
                df.to_csv(output_path, index=False)
                print(f"Saved to {output_path}")
                processed_files.add(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    processed_dir = os.path.join(current_dir, "processed_data")
    
    print("Starting conversion...")
    try:
        process_data(data_dir, processed_dir)
        print("Conversion complete.")
    except ImportError:
        print("Error: pandas or pyreadstat is not installed. Please run: pip install pandas pyreadstat")
    except Exception as e:
        print(f"An error occurred: {e}")
