"""
Akure harmattan visibility: download and completeness check
-----------------------------------------------------------
Source: Iowa Environmental Mesonet (METAR archive), variable vsby.
What it does:
  1. Downloads raw observations for Akure.
  2. Converts visibility from miles to km.
  3. Averages by day first, then by month (avoids reporting-frequency bias).
  4. Assigns each Nov-Mar month to a harmattan season (season year = starting year).
  5. Reports completeness per season and saves CSV files.

Run:  python akure_visibility_check.py
Needs: pip install pandas requests
"""

import io
import os
import calendar
import requests
import pandas as pd

# ---------------- Settings (edit here) ----------------
STATION = "DNAK"        # Akure station ID on IEM (confirm on the IEM site)
START_YEAR = 2010
END_DATE = pd.Timestamp.today().normalize()
MIN_DAY_COVERAGE = 0.70  # a season needs at least 70% of its days with data
OUT_DIR = "outputs"
# ------------------------------------------------------

MILES_TO_KM = 1.609344
HARMATTAN_MONTHS = [11, 12, 1, 2, 3]

os.makedirs(OUT_DIR, exist_ok=True)


def download_raw():
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
    params = {
        "station": STATION,
        "data": "vsby",
        "year1": START_YEAR, "month1": 1, "day1": 1,
        "year2": END_DATE.year, "month2": END_DATE.month, "day2": END_DATE.day,
        "tz": "Africa/Lagos",     # local time so days match Nigerian days
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "trace": "T",
        "direct": "no",
        "report_type": [3, 4],    # routine METAR and SPECI
    }
    print(f"Downloading {STATION} from {START_YEAR} ...")
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    raw_path = os.path.join(OUT_DIR, f"{STATION}_raw_vsby.csv")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"Saved raw file: {raw_path}")
    return raw_path


def load_clean(raw_path):
    df = pd.read_csv(raw_path, na_values=["M", "T"])
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df["vsby"] = pd.to_numeric(df["vsby"], errors="coerce")
    df = df.dropna(subset=["valid", "vsby"])
    df["vis_km"] = df["vsby"] * MILES_TO_KM
    # keep only harmattan months
    df = df[df["valid"].dt.month.isin(HARMATTAN_MONTHS)].copy()
    print(f"Valid harmattan-month observations: {len(df)}")
    print(f"Visibility range (km): {df['vis_km'].min():.2f} to {df['vis_km'].max():.2f}")
    return df


def build_daily(df):
    df["date"] = df["valid"].dt.normalize()
    daily = (
        df.groupby("date")
        .agg(vis_km=("vis_km", "mean"), n_obs=("vis_km", "size"))
        .reset_index()
    )
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    # Season year = year in which the season starts (Nov of that year)
    daily["season"] = daily["year"].where(daily["month"] >= 11, daily["year"] - 1)
    return daily


def season_days(season):
    """Expected days in Nov (year) to Mar (year+1)."""
    feb = 29 if calendar.isleap(season + 1) else 28
    return 30 + 31 + 31 + feb + 31


def completeness(daily):
    rows = []
    for season, g in daily.groupby("season"):
        expected = season_days(season)
        season_end = pd.Timestamp(season + 1, 3, 31)
        finished = END_DATE >= season_end
        rows.append({
            "season": f"{season}/{str(season + 1)[-2:]}",
            "season_start_year": season,
            "days_with_data": len(g),
            "expected_days": expected,
            "coverage_pct": round(100 * len(g) / expected, 1),
            "avg_obs_per_day": round(g["n_obs"].mean(), 1),
            "season_finished": finished,
            "mean_vis_km": round(g["vis_km"].mean(), 2),
        })
    table = pd.DataFrame(rows)
    table["usable"] = (
        (table["coverage_pct"] >= MIN_DAY_COVERAGE * 100) & table["season_finished"]
    )
    return table


def monthly_means(daily):
    m = (
        daily.groupby(["season", "month"])
        .agg(mean_vis_km=("vis_km", "mean"), days_with_data=("vis_km", "size"))
        .reset_index()
    )
    return m


def main():
    raw_path = os.path.join(OUT_DIR, f"{STATION}_raw_vsby.csv")
    if not os.path.exists(raw_path):
        raw_path = download_raw()
    else:
        print(f"Using existing raw file: {raw_path} (delete it to re-download)")

    df = load_clean(raw_path)
    daily = build_daily(df)
    table = completeness(daily)
    monthly = monthly_means(daily)

    daily.to_csv(os.path.join(OUT_DIR, "akure_daily_visibility.csv"), index=False)
    monthly.to_csv(os.path.join(OUT_DIR, "akure_monthly_visibility.csv"), index=False)
    table.to_csv(os.path.join(OUT_DIR, "akure_season_completeness.csv"), index=False)

    print("\nSeason completeness:")
    print(table.to_string(index=False))
    print(f"\nUsable seasons (>= {int(MIN_DAY_COVERAGE * 100)}% days, season finished): "
          f"{int(table['usable'].sum())} of {len(table)}")
    print(f"Files saved in the '{OUT_DIR}' folder.")


if __name__ == "__main__":
    main()
