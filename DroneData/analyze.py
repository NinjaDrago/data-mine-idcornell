import pandas as pd
import json
import os

def main():
    if os.path.exists("uav_logs.csv"):
        df = pd.read_csv("uav_logs.csv")
        print("Loaded CSV data.")
    elif os.path.exists("uav_logs.json"):
        with open("uav_logs.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print("Loaded JSON data.")
    else:
        print("No data file found. Run get_data.py first.")
        return

    print("=== UAV Flight Log Summary ===")
    print(f"Total records: {len(df)}")
    print("\nColumns in dataset:")
    print(df.columns.tolist())

    # Identify date column for summary
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df["YearMonth"] = df[col].dt.to_period("M")
            counts = df["YearMonth"].value_counts().sort_index()
            print("\nFlights per month:")
            print(counts)
            date_col = col
            break

    # Print date, location, and flight_time if they exist
    print("\n=== Detailed Flight Records ===")
    columns_to_show = []
    for col in ["date", "location", "flight_time"]:
        for existing_col in df.columns:
            if col in existing_col.lower():
                columns_to_show.append(existing_col)
                break

    if columns_to_show:
        print(df[columns_to_show].to_string(index=False))
    else:
        print("No date, location, or flight_time columns found.")

if __name__ == "__main__":
    main()
