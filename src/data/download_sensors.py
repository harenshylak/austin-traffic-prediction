"""
download_sensors.py
Downloads Austin radar traffic sensor data from the City of Austin SODA API.

Dataset: Radar Traffic Counts (speed + volume per 15-min interval)
Source:  https://data.austintexas.gov/resource/i626-g7ub

Note: dataset covers 2017-2021. Default range is 2020-2021.

Usage:
    python src/data/download_sensors.py
    python src/data/download_sensors.py --start 2020 --end 2021
"""

import argparse
import os
import sys
import time

import pandas as pd
import requests

API_URL = "https://data.austintexas.gov/resource/i626-g7ub.json"
PAGE_SIZE = 50_000

KEEP_COLS = [
    "detid",
    "int_id",
    "intname",
    "detname",
    "direction",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "volume",
    "speed",
    "occupancy",
]


def fetch_page(year_filter: str, offset: int, timeout: int = 60) -> list[dict]:
    params = {
        "$where": year_filter,
        "$limit": PAGE_SIZE,
        "$offset": offset,
        "$order": "year,month,day,hour,minute,detid",
    }
    resp = requests.get(API_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def download_sensors(start_year: int, end_year: int, out_path: str) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    year_filter = "year IN (" + ",".join(f"'{y}'" for y in years) + ")"

    print(f"Downloading radar traffic counts for years {start_year}-{end_year}...")
    print(f"  API  : {API_URL}")
    print(f"  Filter: {year_filter}")

    chunks = []
    offset = 0
    while True:
        print(f"  Fetching rows {offset:,} – {offset + PAGE_SIZE:,}...", end=" ", flush=True)
        page = fetch_page(year_filter, offset)
        print(f"{len(page)} rows")
        if not page:
            break
        chunks.append(pd.DataFrame(page))
        offset += len(page)
        if len(page) < PAGE_SIZE:
            break
        time.sleep(0.25)  # be a polite API citizen

    if not chunks:
        print("ERROR: No data returned for the requested date range.")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)

    # Keep only relevant columns (drop any missing)
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # Cast numeric columns
    for col in ["year", "month", "day", "hour", "minute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "speed", "occupancy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a datetime column
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour", "minute"]].rename(
            columns={"year": "year", "month": "month", "day": "day",
                     "hour": "hour", "minute": "minute"}
        ),
        errors="coerce",
    )

    # Drop rows where datetime couldn't be parsed
    bad = df["datetime"].isna().sum()
    if bad:
        print(f"  Dropping {bad:,} rows with unparseable timestamps.")
        df = df.dropna(subset=["datetime"])

    df = df.sort_values(["datetime", "detid"]).reset_index(drop=True)

    print(f"\nTotal rows     : {len(df):,}")
    print(f"Unique sensors : {df['detid'].nunique()}")
    print(f"Date range     : {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Missing values :\n{df.isnull().sum().to_string()}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Download Austin radar traffic sensor data")
    parser.add_argument("--start", type=int, default=2020, help="Start year (default: 2020)")
    parser.add_argument("--end",   type=int, default=2021, help="End year (default: 2021)")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw", "radar_traffic.csv"),
        help="Output CSV path (default: data/raw/radar_traffic.csv)",
    )
    args = parser.parse_args()

    if args.start > args.end:
        print("ERROR: --start must be <= --end")
        sys.exit(1)

    download_sensors(args.start, args.end, args.out)


if __name__ == "__main__":
    main()
