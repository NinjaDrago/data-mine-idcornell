import requests
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
        data = r.json()
    except json.JSONDecodeError:
        print("❌ Error: Response is not valid JSON.")
        return

    with open("uav_logs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Download complete. Data saved to uav_logs.json")

if __name__ == "__main__":
    main()
