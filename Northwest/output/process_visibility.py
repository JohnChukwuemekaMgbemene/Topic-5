"""
Topic 5 - Harmattan Visibility Processing Script
Processes DNKN.csv (Kano), DNSO.csv (Sokoto), DNKT.csv (Katsina)
METAR vsby data -> daily mean/min -> monthly mean/min -> descriptive stats + Mann-Kendall

Run from the same directory as the three CSV files.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------- CONFIG ----------
RUN_DIR = Path(__file__).parent
STATIONS = {
    "DNKN": "Kano",
    "DNSO": "Sokoto",
    "DNKT": "Katsina",
}
HARMATTAN_MONTHS = [11, 12, 1, 2, 3]  # Nov, Dec, Jan, Feb, Mar
OUT_DIR = RUN_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load_station_csv(path):
    """Load a raw METAR csv, parse valid datetime, coerce vsby to numeric."""
    df = pd.read_csv(path, dtype=str, sep=None, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    # datetime parsing - try common formats, fall back to flexible parsing
    parsed = pd.to_datetime(df["valid"], format="%Y-%m-%d %H:%M", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        parsed.loc[still_missing] = pd.to_datetime(
            df.loc[still_missing, "valid"], format="%d/%m/%Y %H:%M", errors="coerce"
        )
    still_missing = parsed.isna()
    if still_missing.any():
        # last resort: let pandas infer, dayfirst=False since ISO format is more common in this data
        parsed.loc[still_missing] = pd.to_datetime(
            df.loc[still_missing, "valid"], errors="coerce", dayfirst=False
        )
    df["valid"] = parsed

    # 'null' (and other junk) -> NaN, then numeric
    df["vsby"] = df["vsby"].replace(
        to_replace=["null", "NULL", "Null", "M", ""], value=np.nan
    )
    df["vsby"] = pd.to_numeric(df["vsby"], errors="coerce")

    # drop rows where we couldn't even parse a timestamp
    df = df.dropna(subset=["valid"])

    df["date"] = df["valid"].dt.date
    df["year"] = df["valid"].dt.year
    df["month"] = df["valid"].dt.month

    return df


def assign_harmattan_season(row):
    """
    Harmattan season spans two calendar years (Nov-Mar).
    Label a season by its START year, e.g. Nov 2001 - Mar 2002 => season 2001.
    """
    if row["month"] in (11, 12):
        return row["year"]
    elif row["month"] in (1, 2, 3):
        return row["year"] - 1
    else:
        return np.nan  # not a harmattan month


def daily_aggregation(df):
    """
    Step 1: collapse raw obs (any count, any spacing, gaps included) to one
    row per calendar day: daily mean vsby, daily min vsby, obs count that day.
    """
    daily = (
        df.groupby("date")["vsby"]
        .agg(daily_mean="mean", daily_min="min", n_obs="count")
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["season"] = daily.apply(assign_harmattan_season, axis=1)
    return daily


def monthly_aggregation(daily):
    """
    Step 2: from daily values, build one row per (season, month):
    monthly mean of daily means, monthly mean of daily minimums,
    plus day-coverage info (days with data / days in that month).
    """
    harm = daily.dropna(subset=["season"]).copy()
    harm["season"] = harm["season"].astype(int)

    grouped = harm.groupby(["season", "month"])
    monthly = grouped.agg(
        monthly_mean_vsby=("daily_mean", "mean"),
        monthly_mean_of_daily_min=("daily_min", "mean"),
        days_with_data=("daily_mean", "count"),
        total_obs=("n_obs", "sum"),
    ).reset_index()

    # calendar days in that month (handles Feb leap years using the season's actual year)
    def days_in_month(row):
        # month 11/12 belong to the season's start year; 1/2/3 belong to season+1
        yr = row["season"] if row["month"] in (11, 12) else row["season"] + 1
        return pd.Period(f"{int(yr)}-{int(row['month']):02d}").days_in_month

    monthly["calendar_days"] = monthly.apply(days_in_month, axis=1).astype(int)
    monthly["completeness_pct"] = (
        monthly["days_with_data"] / monthly["calendar_days"] * 100
    ).round(1)

    return monthly


def season_aggregation(monthly):
    """
    Collapse the 5 harmattan months into one row per season (for Mann-Kendall
    input): season mean vsby (mean of monthly means) and season mean of
    monthly-mean-of-daily-min.
    """
    season = monthly.groupby("season").agg(
        season_mean_vsby=("monthly_mean_vsby", "mean"),
        season_mean_min_vsby=("monthly_mean_of_daily_min", "mean"),
        months_present=("month", "count"),  # out of 5 possible (Nov-Mar)
        days_with_data=("days_with_data", "sum"),
        calendar_days=("calendar_days", "sum"),
    ).reset_index()
    season["season_completeness_pct"] = (
        season["days_with_data"] / season["calendar_days"] * 100
    ).round(1)
    return season


def descriptive_stats(monthly):
    """
    Per-month descriptive stats across all seasons: mean, SD, CV,
    and mean-of-daily-minimums, for each of the 5 harmattan months.
    """
    stats = monthly.groupby("month").agg(
        mean_vsby=("monthly_mean_vsby", "mean"),
        sd_vsby=("monthly_mean_vsby", "std"),
        mean_of_min_vsby=("monthly_mean_of_daily_min", "mean"),
        n_seasons=("monthly_mean_vsby", "count"),
    ).reset_index()
    stats["cv_pct"] = (stats["sd_vsby"] / stats["mean_vsby"] * 100).round(1)

    month_names = {11: "November", 12: "December", 1: "January", 2: "February", 3: "March"}
    stats["month_name"] = stats["month"].map(month_names)
    stats = stats.sort_values(
        by="month", key=lambda s: s.map({11: 0, 12: 1, 1: 2, 2: 3, 3: 4})
    )
    return stats


def mann_kendall(x):
    """
    Basic Mann-Kendall trend test + Sen's slope, no external dependency.
    x: 1D array-like of values in time order (NaNs dropped).
    Returns dict with S, Z, p (two-sided, normal approx), trend label, sen_slope.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return {"n": n, "S": np.nan, "Z": np.nan, "p": np.nan,
                "trend": "insufficient data", "sen_slope": np.nan}

    # S statistic
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))

    # variance with tie correction
    unique, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # two-sided p-value from standard normal (no scipy dependency)
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / np.sqrt(2))))

    if p < 0.05:
        trend = "increasing (significant)" if s > 0 else "decreasing (significant)"
    else:
        trend = "no significant trend"

    # Sen's slope: median of all pairwise slopes
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slopes.append((x[j] - x[i]) / (j - i))
    sen_slope = np.median(slopes)

    return {"n": n, "S": s, "Z": round(z, 3), "p": round(p, 4),
            "trend": trend, "sen_slope": round(sen_slope, 4)}


def process_station(code, name):
    csv_path = RUN_DIR / f"{code}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return None

    print(f"\n=== {name} ({code}) ===")
    raw = load_station_csv(csv_path)
    print(f"  raw rows (with usable timestamp): {len(raw)}")
    print(f"  rows with valid vsby: {raw['vsby'].notna().sum()}")
    print(f"  date range: {raw['valid'].min()} to {raw['valid'].max()}")

    daily = daily_aggregation(raw)
    monthly = monthly_aggregation(daily)
    season = season_aggregation(monthly)
    stats = descriptive_stats(monthly)

    mk_mean = mann_kendall(season.sort_values("season")["season_mean_vsby"].values)
    mk_min = mann_kendall(season.sort_values("season")["season_mean_min_vsby"].values)

    print(f"  seasons with any data: {len(season)}")
    print(f"  Mann-Kendall (season mean vsby): {mk_mean}")
    print(f"  Mann-Kendall (season mean of daily min vsby): {mk_min}")

    # save outputs
    daily.to_csv(OUT_DIR / f"{code}_daily.csv", index=False)
    monthly.to_csv(OUT_DIR / f"{code}_monthly.csv", index=False)
    season.to_csv(OUT_DIR / f"{code}_season.csv", index=False)
    stats.to_csv(OUT_DIR / f"{code}_descriptive_stats.csv", index=False)

    # decade-by-decade completeness (season-start-year based decade bucket)
    season_dec = season.copy()
    season_dec["decade"] = (season_dec["season"] // 10) * 10
    decade_completeness = season_dec.groupby("decade").agg(
        seasons_in_decade=("season", "count"),
        avg_completeness_pct=("season_completeness_pct", "mean"),
        avg_season_mean_vsby=("season_mean_vsby", "mean"),
    ).reset_index()
    decade_completeness["avg_completeness_pct"] = decade_completeness["avg_completeness_pct"].round(1)
    decade_completeness["avg_season_mean_vsby"] = decade_completeness["avg_season_mean_vsby"].round(2)
    decade_completeness.to_csv(OUT_DIR / f"{code}_decade_completeness.csv", index=False)
    print(f"  decade completeness:\n{decade_completeness.to_string(index=False)}")

    pd.DataFrame([
        {"metric": "season_mean_vsby", **mk_mean},
        {"metric": "season_mean_of_daily_min_vsby", **mk_min},
    ]).to_csv(OUT_DIR / f"{code}_mann_kendall.csv", index=False)

    return {
        "station": name, "code": code,
        "raw_rows": len(raw), "valid_vsby_rows": int(raw["vsby"].notna().sum()),
        "date_min": raw["valid"].min(), "date_max": raw["valid"].max(),
        "seasons_with_data": len(season),
        "mk_mean": mk_mean, "mk_min": mk_min,
    }


def main():
    summary = []
    for code, name in STATIONS.items():
        result = process_station(code, name)
        if result:
            summary.append(result)

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(OUT_DIR / "station_summary.csv", index=False)
        print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
