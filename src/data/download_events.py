"""
download_events.py
Downloads Austin permitted special events from the City of Austin ACE Events API.
https://data.austintexas.gov/Recreation-and-Culture/ACE-Events/teth-r7k8/about_data

Each permitted event has exact start/end datetimes, tier type (3=large, 4=major),
and road closure information. Multiple rows per event exist (one per road closure
segment) — we deduplicate by folderrsn and aggregate closure info.

Output columns:
  start_dt          : event start datetime (ISO 8601)
  end_dt            : event end datetime
  event_name        : permit folder name
  tier_type         : 3 or 4 (City of Austin tier)
  has_road_closure  : 1 if any full road closure associated
  road_closure_count: number of distinct road segments closed
  amplified_sound   : 1 if amplified sound permitted
  alcohol_served    : 1 if alcohol permit included

Usage:
    python -m src.data.download_events
    python -m src.data.download_events --out data/raw/events.csv
    python -m src.data.download_events --start 2020-04-01 --end 2021-09-30
"""

import argparse
import os
import time

import pandas as pd
import requests

API_URL  = "https://data.austintexas.gov/resource/teth-r7k8.json"
PAGE_SIZE = 50_000


def fetch_events(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download all ACE events whose start_date falls within [start_date, end_date].
    Paginates automatically.
    """
    where = f"start_date >= '{start_date}' AND start_date <= '{end_date}'"
    params = {
        "$where":  where,
        "$select": (
            "folderrsn, foldername, start_date, end_date, tier_type, "
            "road_closure, type_of_road_closure, amplified_sound, alcohol_served"
        ),
        "$limit":  PAGE_SIZE,
        "$offset": 0,
        "$order":  "start_date ASC",
    }

    all_rows = []
    page = 0
    while True:
        params["$offset"] = page * PAGE_SIZE
        print(f"  Fetching page {page + 1} (offset={params['$offset']})...", end=" ")
        resp = requests.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        print(f"{len(rows)} rows")
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break
        page += 1
        time.sleep(0.3)  # polite rate limiting

    return pd.DataFrame(all_rows)


def build_events_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate by folderrsn, aggregate road closure info per event.

    Returns one row per unique permitted event.
    """
    if raw.empty:
        return pd.DataFrame()

    # Parse datetimes
    raw["start_dt"] = pd.to_datetime(raw["start_date"], errors="coerce")
    raw["end_dt"]   = pd.to_datetime(raw["end_date"],   errors="coerce")
    raw["tier_type"] = pd.to_numeric(raw["tier_type"], errors="coerce").fillna(3).astype(int)

    # Boolean flags
    raw["amplified_sound"] = (raw.get("amplified_sound", "No") == "Yes").astype(int)
    raw["alcohol_served"]  = (raw.get("alcohol_served",  "No") == "Yes").astype(int)

    # Road closure: full closure = road_closure IS NOT NULL and type contains 'Full Road'
    raw["is_full_closure"] = (
        raw.get("type_of_road_closure", "").str.contains("Full Road", na=False)
    ).astype(int)

    # Aggregate per unique event (folderrsn)
    agg = (
        raw.groupby("folderrsn")
        .agg(
            event_name        = ("foldername",     "first"),
            start_dt          = ("start_dt",        "first"),
            end_dt            = ("end_dt",           "first"),
            tier_type         = ("tier_type",        "first"),
            has_road_closure  = ("is_full_closure",  "max"),
            road_closure_count= ("road_closure",     lambda x: x.notna().sum()),
            amplified_sound   = ("amplified_sound",  "max"),
            alcohol_served    = ("alcohol_served",   "max"),
        )
        .reset_index(drop=True)
    )

    # Drop events with missing datetimes
    agg = agg.dropna(subset=["start_dt", "end_dt"])

    # Sort by start time
    agg = agg.sort_values("start_dt").reset_index(drop=True)

    return agg


def main():
    parser = argparse.ArgumentParser(description="Download Austin ACE Events 2020-2021")
    parser.add_argument("--out",   default=os.path.join("data", "raw", "events.csv"))
    parser.add_argument("--start", default="2020-04-01")
    parser.add_argument("--end",   default="2021-09-30")
    args = parser.parse_args()

    print(f"Downloading ACE events from {args.start} to {args.end}...")
    raw = fetch_events(args.start, args.end)
    print(f"  Raw rows fetched: {len(raw):,}")

    events = build_events_df(raw)
    print(f"  Unique events after dedup: {len(events):,}")

    if events.empty:
        print("No events found — check date range or API.")
        return

    print(f"\n  Date range: {events['start_dt'].min()} → {events['end_dt'].max()}")
    print(f"  Events with full road closure: {events['has_road_closure'].sum()}")
    print(f"  Tier breakdown:\n{events['tier_type'].value_counts().to_string()}")
    print(f"\n  Sample events:")
    print(events[["event_name", "start_dt", "end_dt", "has_road_closure", "road_closure_count"]].head(10).to_string(index=False))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    events.to_csv(args.out, index=False)
    print(f"\nSaved {len(events)} events to {args.out}")


if __name__ == "__main__":
    main()
