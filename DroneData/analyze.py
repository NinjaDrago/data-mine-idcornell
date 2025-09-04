import pandas as pd
import json

def main():
    try:
        with open("uav_logs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Data file not found. Run get_data.py first to download the dataset.")
        return

    df = pd.DataFrame(data)
    print("=== UAV Flight Log Summary ===")
    print(f"Total records: {len(df)}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())

    # If 'date' column exists, summarize by month
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df["YearMonth"] = df[date_col].dt.to_period("M")
            counts = df["YearMonth"].value_counts().sort_index()
            print("\nFlights per month:")
            print(counts)
            break

    # Try homebrew value counts for potential type/model fields
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["model", "type", "make"]):
            print(f"\nTop values in '{col}':")
            print(df[col].value_counts().head(10))

if __name__ == "__main__":
    main()
