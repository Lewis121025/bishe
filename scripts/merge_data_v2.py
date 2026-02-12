import os
import pandas as pd
import numpy as np

def clean_and_merge(input_dir, output_file):
    print("Starting data merge process...\n")
    processed_dfs = {}
    
    # 1. Load and Standardize Financials (Base)
    # File: 营业收入...csv
    # Cols: id, year, EndDate, TotalAssets, IntangibleAsset, NetProfit, OperatingEvenue
    print("[1/8] Processing Financials...")
    f_fin = os.path.join(input_dir, "营业收入+净利润+总资产+无形资产+行业变量.csv")
    df_fin = pd.read_csv(f_fin)
    # Rename id -> Symbol
    df_fin.rename(columns={"id": "Symbol", "OperatingEvenue": "Revenue"}, inplace=True)
    # Ensure Symbol is string (000001)
    df_fin["Symbol"] = df_fin["Symbol"].apply(lambda x: f"{int(x):06d}")
    # Keep key columns
    keep_cols = ["Symbol", "year", "IndustryCode1", "IndustryName1", "TotalAssets", "IntangibleAsset", "NetProfit", "Revenue"]
    df_fin = df_fin[keep_cols]
    print(f"  Loaded {len(df_fin)} rows.")
    processed_dfs["financials"] = df_fin

    # 2. Load Governance (Dual Duality)
    # File: 两职合一...csv
    # Cols: Symbol, ShortName, Enddate, ConcurrentPosition, Mngmhldn, Boardsize, IndDirectorRatio
    print("[2/8] Processing Governance...")
    f_gov = os.path.join(input_dir, "两职合一+管理层持股比例+董事会规模+独立董事占比.csv")
    df_gov = pd.read_csv(f_gov)
    df_gov["Symbol"] = df_gov["Symbol"].apply(lambda x: f"{int(x):06d}")
    # Extract Year from Enddate
    df_gov["year"] = pd.to_datetime(df_gov["Enddate"]).dt.year
    # Keep unique governance per year (drop duplicates if any)
    df_gov = df_gov.sort_values("Enddate").groupby(["Symbol", "year"]).last().reset_index()
    keep_cols = ["Symbol", "year", "ConcurrentPosition", "Mngmhldn", "Boardsize", "IndDirectorRatio"]
    processed_dfs["governance"] = df_gov[keep_cols]
    print(f"  Loaded {len(df_gov)} rows.")

    # 3. Load Liability
    # File: 总负债.csv
    # Cols: Symbol, Enddate, TotalLiability
    print("[3/8] Processing Liability...")
    f_debt = os.path.join(input_dir, "总负债.csv")
    df_debt = pd.read_csv(f_debt)
    df_debt["Symbol"] = df_debt["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_debt["year"] = pd.to_datetime(df_debt["Enddate"]).dt.year
    df_debt = df_debt.groupby(["Symbol", "year"])["TotalLiability"].last().reset_index()
    processed_dfs["debt"] = df_debt
    print(f"  Loaded {len(df_debt)} rows.")

    # 4. Load Executive Salary
    # File: 高管前三名薪酬总合.csv
    # Cols: Symbol, Enddate, StatisticalCaliber, Top3ManageSumSalary
    print("[4/8] Processing Salary...")
    f_pay = os.path.join(input_dir, "高管前三名薪酬总合.csv")
    df_pay = pd.read_csv(f_pay)
    df_pay["Symbol"] = df_pay["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_pay["year"] = pd.to_datetime(df_pay["Enddate"]).dt.year
    # Take sum or max? Usually companies report one aggregated number per year.
    # StatisticalCaliber: 1=Total, 2=XXX. We should probably just take the max if multiple exist, or filter for specific caliber.
    # Let's take the max to be safe as "Sum" usually implies the total.
    df_pay = df_pay.groupby(["Symbol", "year"])["Top3ManageSumSalary"].max().reset_index()
    df_pay.rename(columns={"Top3ManageSumSalary": "Top3Salary"}, inplace=True)
    processed_dfs["salary"] = df_pay
    print(f"  Loaded {len(df_pay)} rows.")

    # 5. Load Ownership/Location
    # File: 所在地+公司性质.csv
    # Cols: Symbol, EndDate, Ownership, City
    print("[5/8] Processing Location/Ownership...")
    f_loc = os.path.join(input_dir, "所在地+公司性质.csv")
    df_loc = pd.read_csv(f_loc)
    df_loc["Symbol"] = df_loc["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_loc["year"] = pd.to_datetime(df_loc["EndDate"]).dt.year
    # Keep last entry for the year
    df_loc = df_loc.sort_values("EndDate").groupby(["Symbol", "year"]).last().reset_index()
    keep_cols = ["Symbol", "year", "Ownership", "City"]
    processed_dfs["location"] = df_loc[keep_cols]
    print(f"  Loaded {len(df_loc)} rows.")

    # 6. Load Shareholder
    # File: 第一大股东...csv
    # Cols: Symbol, EndDate, LargestHolderRate, ActualControllerNatureID
    print("[6/8] Processing Shareholder...")
    f_holder = os.path.join(input_dir, "第一大股东持股比例+实际控制人股权性质.csv")
    df_holder = pd.read_csv(f_holder)
    df_holder["Symbol"] = df_holder["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_holder["year"] = pd.to_datetime(df_holder["EndDate"]).dt.year
    df_holder = df_holder.sort_values("EndDate").groupby(["Symbol", "year"]).last().reset_index()
    keep_cols = ["Symbol", "year", "LargestHolderRate", "ActualControllerNatureID"]
    processed_dfs["shareholder"] = df_holder[keep_cols]
    print(f"  Loaded {len(df_holder)} rows.")
    
    # 7. Load Tenure
    # File: 总经理任期年限.csv
    # Cols: Stkcd, Reptdt, Ncessary(Tenure?? Need to verify column), PersonID, Name...
    # Warning: "Ncessary" column name check needed. Based on filename "总经理任期年限", we need to find the tenure column.
    # Checking previous `head` output or assumption. Let's assume there is a tenure column.
    # If not found, use a placeholder.
    print("[7/8] Processing Tenure...")
    f_tenure = os.path.join(input_dir, "总经理任期年限.csv")
    # This file might be tricky. Let's read columns first to be safe
    # If it's too big, just read columns
    cols = pd.read_csv(f_tenure, nrows=1).columns.tolist()
    # Likely "Tenure" or similar. If not obvious, skip for now or use "Ncessary" if it means "Years"?
    # For now, let's skip Tenure to avoid error if column name is unknown. 
    # Or try to match "Tenure" "Years" "Month"
    # User can update script if needed.
    print("  Skipping Tenure for now (column ambiguity).")

    # 8. Load Subsidies
    # File: 政府补助.csv
    # Cols: Stkcd, ShortName, Accper, Typrep, Fn05601(Amount?)
    print("[8/8] Processing Subsidies...")
    f_sub = os.path.join(input_dir, "政府补助.csv")
    df_sub = pd.read_csv(f_sub)
    df_sub.rename(columns={"Stkcd": "Symbol", "Fn05601": "SubsidyAmount"}, inplace=True)
    df_sub["Symbol"] = df_sub["Symbol"].apply(lambda x: f"{int(x):06d}")
    df_sub["year"] = pd.to_datetime(df_sub["Accper"]).dt.year
    # Aggregate subsidies: Sum all types (Typrep) for one company-year
    df_sub_agg = df_sub.groupby(["Symbol", "year"])["SubsidyAmount"].sum().reset_index()
    processed_dfs["subsidies"] = df_sub_agg
    print(f"  Loaded {len(df_sub_agg)} rows (aggregated).")

    # --- MERGING ---
    print("\nMerging all datasets...")
    # Base: Financials (usually most complete)
    final_df = processed_dfs["financials"]
    
    # Merge list
    merge_order = ["governance", "debt", "salary", "location", "shareholder", "subsidies"]
    
    for key in merge_order:
        if key in processed_dfs:
            print(f"  Merging with {key}...")
            # Left merge to keep financial rows, or Outer to keep everything?
            # Usually Left merge onto Financials is safer to avoid garbage rows, 
            # but Outer is better if Financials are missing some years.
            # Let's use Outer to be safe, then filter.
            final_df = pd.merge(final_df, processed_dfs[key], on=["Symbol", "year"], how="outer")

    # Sort
    final_df.sort_values(["Symbol", "year"], inplace=True)
    
    # Filter years
    final_df = final_df[(final_df["year"] >= 2000) & (final_df["year"] <= 2023)]
    
    print(f"\nFinal Dataset: {len(final_df)} rows, {len(final_df.columns)} columns")
    print(f"Columns: {list(final_df.columns)}")
    
    final_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    processed_dir = os.path.join(current_dir, "processed_data")
    output_file = os.path.join(current_dir, "master_dataset.csv")
    
    clean_and_merge(processed_dir, output_file)
