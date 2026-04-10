"""
download_weather.py
Downloads historical hourly weather data for Austin, TX from Open-Meteo.
Saves to data/raw/weather.csv relative to the project root.

Usage:
    python src/data/download_weather.py
    python src/data/download_weather.py --out data/raw/weather.csv
"""

import argparse
import json
import os
import sys
import requests
import pandas as pd


AUSTIN_LAT = 30.2672
AUSTIN_LON = -97.7431

API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": AUSTIN_LAT,
    "longitude": AUSTIN_LON,
    "start_date": "2020-01-01",
    "end_date": "2021-12-31",
    "hourly": ",".join([
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "relative_humidity_2m",
        "visibility",
        "weather_code",
    ]),
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "timezone": "America/Chicago",  # Austin local time
}


def download_weather(out_path: str) -> pd.DataFrame:
    print(f"Requesting weather data from Open-Meteo...")
    print(f"  Location : Austin, TX ({AUSTIN_LAT}, {AUSTIN_LON})")
    print(f"  Period   : {PARAMS['start_date']} to {PARAMS['end_date']}")
    print(f"  Variables: {PARAMS['hourly']}")

    response = requests.get(API_URL, params=PARAMS, timeout=60)
    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        print("ERROR: Unexpected response structure:")
        print(json.dumps(data, indent=2)[:500])
        sys.exit(1)

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)

    # Parse and set datetime index (Open-Meteo returns ISO strings)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    df.index.name = "datetime"

    # Rename columns for clarity
    df = df.rename(columns={
        "temperature_2m":       "temp_f",
        "precipitation":        "precip_in",
        "wind_speed_10m":       "wind_mph",
        "relative_humidity_2m": "humidity_pct",
        "visibility":           "visibility_m",
        "weather_code":         "weather_code",
    })

    print(f"\nDownloaded {len(df):,} hourly rows")
    print(f"Date range : {df.index.min()} to {df.index.max()}")
    print(f"Missing values:\n{df.isnull().sum().to_string()}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Austin weather data from Open-Meteo")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw", "weather.csv"),
        help="Output CSV path (default: data/raw/weather.csv)",
    )
    args = parser.parse_args()
    download_weather(args.out)


if __name__ == "__main__":
    main()
