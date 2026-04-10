"""
download_events.py
Builds an Austin special-events calendar for 2020-2021.

Sources:
  - Curated dictionary of major recurring Austin events (SXSW, ACL, F1, UT football, holidays)
  - Note: The City of Austin ACE Events open dataset is no longer publicly available.

Output columns:
  date        : YYYY-MM-DD (one row per calendar day the event is active)
  event_name  : human-readable name
  event_type  : festival | sports | holiday | concert
  impact      : 1 (local), 2 (citywide), 3 (regional/national draw)

Usage:
    python src/data/download_events.py
    python src/data/download_events.py --out data/raw/events.csv
"""

import argparse
import os
from datetime import date, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# Curated event definitions
# Each entry: (name, type, impact, start_date, end_date)
# ---------------------------------------------------------------------------
EVENTS: list[tuple[str, str, int, date, date]] = [

    # --- 2020 ---
    # SXSW 2020 was officially cancelled 2020-03-06 but generated some traffic
    # the days before cancellation; omit to avoid noise.

    # ACL Music Festival 2020 — cancelled (COVID)
    # F1 US Grand Prix 2020 — cancelled (COVID)

    # UT Football 2020 (COVID: limited fans, mostly home bubble)
    ("UT Football vs TCU",         "sports",  1, date(2020, 9, 26), date(2020, 9, 26)),
    ("UT Football vs Texas Tech",  "sports",  1, date(2020, 10, 24), date(2020, 10, 24)),
    ("UT Football vs Kansas",      "sports",  1, date(2020, 11, 7),  date(2020, 11, 7)),
    ("UT Football vs West Virginia","sports", 1, date(2020, 11, 28), date(2020, 11, 28)),

    # Federal holidays 2020
    ("New Year's Day",             "holiday", 1, date(2020, 1, 1),   date(2020, 1, 1)),
    ("MLK Day",                    "holiday", 1, date(2020, 1, 20),  date(2020, 1, 20)),
    ("Memorial Day",               "holiday", 1, date(2020, 5, 25),  date(2020, 5, 25)),
    ("Independence Day",           "holiday", 2, date(2020, 7, 4),   date(2020, 7, 4)),
    ("Labor Day",                  "holiday", 1, date(2020, 9, 7),   date(2020, 9, 7)),
    ("Thanksgiving",               "holiday", 2, date(2020, 11, 26), date(2020, 11, 27)),
    ("Christmas",                  "holiday", 2, date(2020, 12, 25), date(2020, 12, 25)),

    # --- 2021 ---
    # SXSW 2021 — online only, no in-person
    # ACL Music Festival 2021 — two weekends, returned after COVID gap
    ("ACL Music Festival Weekend 1", "festival", 3, date(2021, 10, 1),  date(2021, 10, 3)),
    ("ACL Music Festival Weekend 2", "festival", 3, date(2021, 10, 8),  date(2021, 10, 10)),

    # Formula 1 US Grand Prix 2021
    ("F1 US Grand Prix",           "festival", 3, date(2021, 10, 21), date(2021, 10, 24)),

    # UT Football 2021 home games (Darrell K Royal Stadium)
    ("UT Football vs Louisiana",   "sports",  2, date(2021, 9, 4),   date(2021, 9, 4)),
    ("UT Football vs Rice",        "sports",  2, date(2021, 9, 18),  date(2021, 9, 18)),
    ("UT Football vs Texas Tech",  "sports",  2, date(2021, 9, 25),  date(2021, 9, 25)),
    ("UT Football vs Oklahoma St", "sports",  2, date(2021, 10, 16), date(2021, 10, 16)),
    ("UT Football vs Kansas",      "sports",  2, date(2021, 11, 13), date(2021, 11, 13)),
    ("UT Football vs Kansas St",   "sports",  2, date(2021, 11, 27), date(2021, 11, 27)),

    # Austin FC inaugural MLS season — home games at Q2 Stadium
    ("Austin FC vs LA Galaxy",     "sports",  2, date(2021, 6, 19),  date(2021, 6, 19)),
    ("Austin FC vs Portland",      "sports",  2, date(2021, 7, 3),   date(2021, 7, 3)),
    ("Austin FC vs Houston Dynamo","sports",  2, date(2021, 7, 17),  date(2021, 7, 17)),
    ("Austin FC vs Colorado",      "sports",  2, date(2021, 7, 24),  date(2021, 7, 24)),
    ("Austin FC vs FC Dallas",     "sports",  2, date(2021, 8, 21),  date(2021, 8, 21)),
    ("Austin FC vs Sporting KC",   "sports",  2, date(2021, 9, 11),  date(2021, 9, 11)),
    ("Austin FC vs Vancouver",     "sports",  2, date(2021, 9, 22),  date(2021, 9, 22)),
    ("Austin FC vs Real Salt Lake","sports",  2, date(2021, 10, 2),  date(2021, 10, 2)),
    ("Austin FC vs Minnesota",     "sports",  2, date(2021, 10, 30), date(2021, 10, 30)),

    # Federal holidays 2021
    ("New Year's Day",             "holiday", 1, date(2021, 1, 1),   date(2021, 1, 1)),
    ("MLK Day",                    "holiday", 1, date(2021, 1, 18),  date(2021, 1, 18)),
    ("Presidents Day",             "holiday", 1, date(2021, 2, 15),  date(2021, 2, 15)),
    ("Memorial Day",               "holiday", 1, date(2021, 5, 31),  date(2021, 5, 31)),
    ("Independence Day",           "holiday", 2, date(2021, 7, 4),   date(2021, 7, 4)),
    ("Labor Day",                  "holiday", 1, date(2021, 9, 6),   date(2021, 9, 6)),
    ("Thanksgiving",               "holiday", 2, date(2021, 11, 25), date(2021, 11, 26)),
    ("Christmas",                  "holiday", 2, date(2021, 12, 25), date(2021, 12, 25)),

    # Winter Storm Uri — major traffic disruption
    ("Winter Storm Uri",           "holiday", 3, date(2021, 2, 10),  date(2021, 2, 20)),
]


def build_events_df(events: list[tuple]) -> pd.DataFrame:
    rows = []
    for name, etype, impact, start, end in events:
        current = start
        while current <= end:
            rows.append({
                "date":       current.isoformat(),
                "event_name": name,
                "event_type": etype,
                "impact":     impact,
            })
            current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "event_name"]).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Build Austin events calendar 2020-2021")
    parser.add_argument(
        "--out",
        default=os.path.join("data", "raw", "events.csv"),
        help="Output CSV path (default: data/raw/events.csv)",
    )
    args = parser.parse_args()

    df = build_events_df(EVENTS)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Built events calendar: {len(df)} event-days across {df['date'].nunique()} dates")
    print(f"Event types: {df.groupby('event_type')['date'].count().to_dict()}")
    print(f"Impact distribution: {df['impact'].value_counts().sort_index().to_dict()}")
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
