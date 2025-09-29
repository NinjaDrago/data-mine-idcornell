import requests
import pandas as pd
import json

# OData API endpoint for UAV flight log dataset
URL = "https://data.bloomington.in.gov/resource/3a7f-6kb4.json"

def main():
    print("Downloading UAV flight log data via API...")
    headers = {"Accept": "application/json"}
    r = requests.get(URL, headers=headers)
    
    try:
        r.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return

    try:
        data = r.json()  # Load JSON data
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        return

    # Save JSON
    with open("uav_logs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ Data saved as uav_logs.json")

    # Convert to DataFrame and save CSV
    df = pd.DataFrame(data)
    df.to_csv("uav_logs.csv", index=False)
    print("✅ Data saved as uav_logs.csv")

if __name__ == "__main__":
    main()
