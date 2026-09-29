"""
STEP 2: Build daily, monthly and seasonal dust series and pair them with visibility
-----------------------------------------------------------------------------------
Inputs (in the 'outputs' folder):
  akure_dust_hourly.csv        from step 1 (hourly, UTC)
  akure_daily_visibility.csv   from your visibility check script (daily, local time)
Outputs:
  akure_dust_daily.csv, akure_dust_monthly.csv, akure_dust_seasonal.csv
  akure_dust_visibility_pairs.csv   (days that have both dust and visibility)
"""

import os
import calendar
import numpy as np
import pandas as pd

# ---------------- Settings ----------------
OUT_DIR = "outputs"
SEASON_MONTHS = [11, 12, 1, 2, 3]
UTC_TO_LOCAL_HOURS = 1        # Nigeria is UTC+1
MIN_HOURS_PER_DAY = 20        # a dust day needs at least 20 of 24 hourly values
MIN_MONTH_COVERAGE = 0.80     # a month needs at least 80% of its days
MIN_SEASON_COVERAGE = 0.90    # a season needs at least 90% of its days
VIS_CAP_KM = 10.0             # METAR visibility is normally capped near 10 km
MIN_OBS_PER_DAY = 2           # a visibility day needs at least 2 reports
# ------------------------------------------


def season_days(season):
    feb = 29 if calendar.isleap(season + 1) else 28
    return 30 + 31 + 31 + feb + 31


def main():
    hourly = pd.read_csv(os.path.join(OUT_DIR, "akure_dust_hourly.csv"),
                         parse_dates=["time_utc"])
    print(f"Hourly dust rows read: {len(hourly)}")

    # Hourly to local daily (average by day first)
    hourly["time_local"] = hourly["time_utc"] + pd.Timedelta(hours=UTC_TO_LOCAL_HOURS)
    hourly["date"] = hourly["time_local"].dt.normalize()
    daily = (hourly.groupby("date")
             .agg(dust_ugm3=("dust_ugm3", "mean"),
                  dust_aod=("dust_aod", "mean"),
                  n_hours=("dust_ugm3", "size"))
             .reset_index())
    daily = daily[daily["n_hours"] >= MIN_HOURS_PER_DAY].copy()
    daily["month"] = daily["date"].dt.month
    daily["year"] = daily["date"].dt.year
    daily = daily[daily["month"].isin(SEASON_MONTHS)].copy()
    daily["season"] = daily["year"].where(daily["month"] >= 11, daily["year"] - 1)

    # Daily to monthly (keep only months with enough days)
    monthly = (daily.groupby(["season", "month"])
               .agg(dust_ugm3=("dust_ugm3", "mean"),
                    dust_aod=("dust_aod", "mean"),
                    days=("dust_ugm3", "size"))
               .reset_index())
    monthly["year"] = monthly["season"].where(monthly["month"] >= 11, monthly["season"] + 1)
    monthly["days_in_month"] = [calendar.monthrange(y, m)[1]
                                for y, m in zip(monthly["year"], monthly["month"])]
    monthly["coverage"] = monthly["days"] / monthly["days_in_month"]
    monthly = monthly[monthly["coverage"] >= MIN_MONTH_COVERAGE].copy()

    # Daily to seasonal (keep only seasons with enough days)
    seasonal = (daily.groupby("season")
                .agg(dust_ugm3=("dust_ugm3", "mean"),
                     dust_aod=("dust_aod", "mean"),
                     days=("dust_ugm3", "size"))
                .reset_index())
    seasonal["expected_days"] = seasonal["season"].apply(season_days)
    seasonal["coverage"] = seasonal["days"] / seasonal["expected_days"]
    seasonal = seasonal[seasonal["coverage"] >= MIN_SEASON_COVERAGE].copy()
    seasonal["label"] = seasonal["season"].astype(str) + "/" + \
        (seasonal["season"] + 1).astype(str).str[-2:]

    print(f"Daily dust values kept: {len(daily)}")
    print(f"Months kept: {len(monthly)} | Seasons kept: {len(seasonal)}")

    # Pair with observed visibility
    vis_path = os.path.join(OUT_DIR, "akure_daily_visibility.csv")
    pairs = pd.DataFrame()
    if os.path.exists(vis_path):
        vis = pd.read_csv(vis_path, parse_dates=["date"])
        vis = vis[vis["n_obs"] >= MIN_OBS_PER_DAY].copy()
        n_capped = int((vis["vis_km"] > VIS_CAP_KM).sum())
        vis["vis_km"] = vis["vis_km"].clip(upper=VIS_CAP_KM)
        print(f"Visibility days kept: {len(vis)} ({n_capped} daily means were above "
              f"{VIS_CAP_KM} km and were capped)")
        pairs = pd.merge(daily[["date", "season", "month", "dust_ugm3", "dust_aod"]],
                         vis[["date", "vis_km", "n_obs"]], on="date", how="inner")
        print(f"Days with both dust and visibility: {len(pairs)}")
    else:
        print("akure_daily_visibility.csv not found, so no pairing was done.")

    daily.to_csv(os.path.join(OUT_DIR, "akure_dust_daily.csv"), index=False)
    monthly.to_csv(os.path.join(OUT_DIR, "akure_dust_monthly.csv"), index=False)
    seasonal.to_csv(os.path.join(OUT_DIR, "akure_dust_seasonal.csv"), index=False)
    pairs.to_csv(os.path.join(OUT_DIR, "akure_dust_visibility_pairs.csv"), index=False)
    print("Saved all processed files in the outputs folder.")


if __name__ == "__main__":
    main()
