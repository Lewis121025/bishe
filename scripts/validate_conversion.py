import os
import pandas as pd
import glob

def validate_data(input_dir, output_dir):
    print("Starting validation...\n")
    
    # 1. Map input files to logical groups (handling split files)
    dta_files = glob.glob(os.path.join(input_dir, "*.dta"))
    file_groups = {}
    
    for f in dta_files:
        basename = os.path.basename(f)
        if "两个文件上下拼接" in basename:
            # e.g. "总经理任期年限（两个文件上下拼接）-1.dta" -> "总经理任期年限"
            key = basename.split("（")[0]
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(f)
        else:
            # e.g. "总负债.dta" -> "总负债"
            key = os.path.splitext(basename)[0]
            file_groups[key] = [f]

    # 2. Iterate through groups and validate against output
    for key, input_list in file_groups.items():
        print(f"Checking dataset: {key}")
        
        # Calculate expected metrics from input
        total_rows_input = 0
        input_columns = set()
        
        try:
            for f in input_list:
                df_in = pd.read_stata(f)
                total_rows_input += len(df_in)
                if not input_columns:
                    input_columns = set(df_in.columns)
                else:
                    # check consistency of columns in split files
                    if set(df_in.columns) != input_columns:
                        print(f"  [WARNING] Column mismatch in split files for {key}!")
        except Exception as e:
            print(f"  [ERROR] Failed to read input file {f}: {e}")
            continue

        # Check output
        output_file = os.path.join(output_dir, key + ".csv")
        if not os.path.exists(output_file):
            print(f"  [ERROR] Missing output file: {output_file}")
            continue
            
        try:
            df_out = pd.read_csv(output_file)
            rows_out = len(df_out)
            cols_out = set(df_out.columns)
            
            # Compare
            if rows_out == total_rows_input:
                print(f"  [OK] Row count matches: {rows_out}")
            else:
                print(f"  [FAIL] Row count mismatch! Expected {total_rows_input}, got {rows_out}")
                
            if input_columns == cols_out:
                print(f"  [OK] Column names match.")
            else:
                 # It's possible CSV conversion changed column names slightly (e.g. valid chars), check set difference
                 missing = input_columns - cols_out
                 extra = cols_out - input_columns
                 if missing or extra:
                     print(f"  [WARNING] Columns differ slightly (encoding/naming?).")
                     if missing: print(f"    Missing in CSV: {missing}")
                     if extra: print(f"    Extra in CSV: {extra}")

        except Exception as e:
            print(f"  [ERROR] Failed to read output file {output_file}: {e}")
        
        print("-" * 30)

if __name__ == "__main__":
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, "data")
    processed_dir = os.path.join(current_dir, "processed_data")
    
    validate_data(data_dir, processed_dir)
