"""
STEP 1: Download MERRA-2 hourly dust for the grid cell nearest Akure

Dataset  : MERRA-2 M2T1NXAER (hourly aerosol diagnostics), NASA GES DISC

Variables: DUSMASS (dust surface mass concentration, kg/m3)
           DUEXTTAU (dust extinction optical thickness at 550 nm)

Output   : outputs/akure_dust_hourly.csv

Run test:
    python 01_download_merra2_dust.py test

Run full download:
    python 01_download_merra2_dust.py
"""

import os
import re
import sys
import time
import calendar
import datetime as dt
from urllib.parse import quote

import pandas as pd
import xarray as xr


# ---------------- Settings ----------------

LAT, LON = 7.15, 5.18

FIRST_SEASON = 1984
LAST_SEASON = 2025

SEASON_MONTHS = [11, 12, 1, 2, 3]

VARS = ["DUSMASS", "DUEXTTAU"]

SHORT_NAME = "M2T1NXAER"
VERSION = "5.12.4"

OUT_DIR = "outputs"
CACHE_DIR = os.path.join(OUT_DIR, "dust_cache")

RETRIES = 3

# ------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)


# Ask the server for just these variables
CONSTRAINT = (
    "?dap4.ce="
    + ";".join(f"/{v}" for v in VARS + ["lat", "lon", "time"])
    .replace(";", "%3B")
)


# ---------- Login and session ----------

def get_token():
    """
    Get Earthdata token.

    Order:
    1. EARTHDATA_TOKEN environment variable
    2. earthdata_token.txt
    3. Ask user once
    """

    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()

    if not tok and os.path.exists("earthdata_token.txt"):
        tok = open(
            "earthdata_token.txt",
            encoding="utf-8"
        ).read().strip()

    if not tok:
        tok = input(
            "Paste your Earthdata token and press Enter: "
        ).strip()

        with open(
            "earthdata_token.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(tok)

        print(
            "Token saved in earthdata_token.txt "
            "(keep this file private).",
            flush=True
        )

    return tok


def make_session(token):
    try:
        from pydap.net import create_session

        return create_session(
            session_kwargs={"token": token}
        )

    except Exception:
        import requests

        s = requests.Session()
        s.headers.update(
            {"Authorization": f"Bearer {token}"}
        )

        return s


# ---------- Finding the files ----------

def opendap_url(granule):
    """Build the Cloud OPeNDAP address of one daily MERRA-2 file."""

    umm = granule["umm"]

    for u in umm.get("RelatedUrls", []):

        link = u.get("URL", "")

        if "opendap.earthdata.nasa.gov" in link:

            for tail in (
                ".dmr.html",
                ".dmr",
                ".html"
            ):
                if link.endswith(tail):
                    link = link[:-len(tail)]

            return link

    # Fallback
    cid = granule["meta"]["collection-concept-id"]
    ur = umm["GranuleUR"]

    return (
        f"https://opendap.earthdata.nasa.gov/"
        f"collections/{cid}/granules/"
        f"{quote(ur, safe='')}"
    )


def granule_date(granule):

    m = re.search(
        r"\.(\d{8})\.nc4",
        granule["umm"]["GranuleUR"]
    )

    return (
        dt.datetime.strptime(
            m.group(1),
            "%Y%m%d"
        ).date()
        if m else None
    )


def find_granules(start, end):

    import earthaccess

    res = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        temporal=(
            dt.datetime.combine(
                start,
                dt.time(0, 0)
            ),
            dt.datetime.combine(
                end,
                dt.time(23, 59, 59)
            )
        ),
    )

    out = {}

    for g in res:

        d = granule_date(g)

        if d is not None and start <= d <= end:
            out[d] = g

    return out


# ---------- Reading one day ----------

def extract_point(ds):
    """
    Take the grid cell nearest Akure
    and return an hourly table.
    """

    i = int(
        abs(ds["lat"].values - LAT).argmin()
    )

    j = int(
        abs(ds["lon"].values - LON).argmin()
    )

    grid_lat = float(ds["lat"].values[i])
    grid_lon = float(ds["lon"].values[j])

    sub = (
        ds[VARS]
        .isel(lat=i, lon=j)
        .load()
    )

    df = (
        sub
        .to_dataframe()
        .reset_index()
    )

    out = pd.DataFrame({
        "time_utc": pd.to_datetime(
            df["time"].astype(str)
        ),

        "dust_ugm3": (
            df["DUSMASS"] * 1e9
        ),

        "dust_aod": df["DUEXTTAU"],
    })

    out["grid_lat"] = grid_lat
    out["grid_lon"] = grid_lon

    return out


def fetch_day(granule, session):

    url = opendap_url(granule) + CONSTRAINT

    last_err = None

    for attempt in range(RETRIES):

        try:

            with xr.open_dataset(
                url,
                engine="pydap",
                session=session
            ) as ds:

                return extract_point(ds), None

        except Exception as e:

            last_err = e

            if attempt < RETRIES - 1:

                wait = 2 * (attempt + 1)

                print(
                    f" retry {attempt + 1}/{RETRIES - 1} "
                    f"in {wait}s...",
                    end="",
                    flush=True
                )

                time.sleep(wait)

    return None, last_err


# ---------- Test mode ----------

def test_run():

    day = dt.date(2019, 12, 15)

    print(
        f"Test: looking for the {day} file ...",
        flush=True
    )

    granules = find_granules(day, day)

    if day not in granules:

        print(
            "Could not find that file in the NASA catalogue "
            "(search returned nothing).",
            flush=True
        )

        return

    g = granules[day]

    print(
        "Address used:",
        opendap_url(g),
        flush=True
    )

    session = make_session(get_token())

    print(
        "Requesting Akure grid cell from NASA...",
        flush=True
    )

    t0 = time.time()

    df, err = fetch_day(g, session)

    elapsed = time.time() - t0

    if df is None:

        print(
            f"FAILED after {elapsed:.1f}s. "
            f"Last error: {err}",
            flush=True
        )

        return

    print(
        f"Success ({elapsed:.1f}s)",
        flush=True
    )

    print(
        df.head(24).to_string(index=False),
        flush=True
    )

    print(
        f"\nGrid cell used: "
        f"lat {df['grid_lat'][0]}, "
        f"lon {df['grid_lon'][0]}",
        flush=True
    )

    print(
        "\nTest passed. "
        "Now run without 'test' for the full download.",
        flush=True
    )


# ---------- Full run ----------

def full_run():

    latest_ok = (
        dt.date.today()
        - dt.timedelta(days=35)
    )

    session = make_session(get_token())

    missing = []

    total_months = 0
    completed_months = 0
    total_days_downloaded = 0

    # Count months in the requested seasons
    for season in range(
        FIRST_SEASON,
        LAST_SEASON + 1
    ):
        total_months += len(SEASON_MONTHS)

    print("=" * 70, flush=True)
    print("MERRA-2 AKURE DUST DOWNLOAD", flush=True)
    print("=" * 70, flush=True)

    print(
        f"Seasons       : {FIRST_SEASON}-{LAST_SEASON}",
        flush=True
    )

    print(
        f"Months/season : {', '.join(map(str, SEASON_MONTHS))}",
        flush=True
    )

    print(
        f"Akure point   : {LAT}, {LON}",
        flush=True
    )

    print(
        f"Output folder : {OUT_DIR}",
        flush=True
    )

    print(
        f"Latest allowed: {latest_ok}",
        flush=True
    )

    print("=" * 70, flush=True)

    overall_start = time.time()

    for season in range(
        FIRST_SEASON,
        LAST_SEASON + 1
    ):

        for month in SEASON_MONTHS:

            year = (
                season
                if month >= 11
                else season + 1
            )

            cache = os.path.join(
                CACHE_DIR,
                f"dust_{year}_{month:02d}.csv"
            )

            # Already downloaded
            if os.path.exists(cache):

                print(
                    f"\n[{year}-{month:02d}] "
                    f"Already downloaded — skipping.",
                    flush=True
                )

                completed_months += 1

                continue

            ndays = calendar.monthrange(
                year,
                month
            )[1]

            start = dt.date(
                year,
                month,
                1
            )

            end = min(
                dt.date(
                    year,
                    month,
                    ndays
                ),
                latest_ok
            )

            if end < start:

                print(
                    f"\n[{year}-{month:02d}] "
                    f"Future/unavailable — skipping.",
                    flush=True
                )

                continue

            print("\n" + "=" * 70, flush=True)

            print(
                f"[{year}-{month:02d}] "
                f"Starting month",
                flush=True
            )

            print(
                f"[{year}-{month:02d}] "
                f"Searching NASA catalogue "
                f"({start} → {end})...",
                flush=True
            )

            search_start = time.time()

            try:

                granules = find_granules(
                    start,
                    end
                )

            except Exception as e:

                print(
                    f"[{year}-{month:02d}] "
                    f"CATALOGUE SEARCH FAILED: {e}",
                    flush=True
                )

                continue

            search_time = (
                time.time()
                - search_start
            )

            print(
                f"[{year}-{month:02d}] "
                f"Found {len(granules)} daily files "
                f"in {search_time:.1f}s",
                flush=True
            )

            if not granules:

                print(
                    f"[{year}-{month:02d}] "
                    f"No files found.",
                    flush=True
                )

                continue

            frames = []

            fails = 0

            days = sorted(granules)

            month_start = time.time()

            for n, d in enumerate(days, 1):

                print(
                    f"[{year}-{month:02d}] "
                    f"Day {n}/{len(days)} "
                    f"({d}) ...",
                    end=" ",
                    flush=True
                )

                t0 = time.time()

                df, err = fetch_day(
                    granules[d],
                    session
                )

                elapsed = (
                    time.time()
                    - t0
                )

                if df is None:

                    print(
                        f"FAILED after "
                        f"{elapsed:.1f}s",
                        flush=True
                    )

                    missing.append(str(d))

                    fails += 1

                    if fails >= 5 and not frames:

                        print(
                            "\nFive failures in a row "
                            "with no successful downloads.",
                            flush=True
                        )

                        print(
                            "Last error:",
                            err,
                            flush=True
                        )

                        print(
                            "Stopping. Run test mode "
                            "and send me the error.",
                            flush=True
                        )

                        return

                else:

                    frames.append(df)

                    total_days_downloaded += 1

                    print(
                        f"OK ({elapsed:.1f}s)",
                        flush=True
                    )

            # Save month
            if frames:

                month_df = pd.concat(
                    frames,
                    ignore_index=True
                )

                month_df.to_csv(
                    cache,
                    index=False
                )

                month_elapsed = (
                    time.time()
                    - month_start
                )

                print(
                    f"[{year}-{month:02d}] "
                    f"SAVED: {len(frames)} "
                    f"of {len(days)} days "
                    f"({len(month_df)} hourly rows)",
                    flush=True
                )

                print(
                    f"[{year}-{month:02d}] "
                    f"Month completed in "
                    f"{month_elapsed / 60:.1f} minutes",
                    flush=True
                )

            else:

                print(
                    f"[{year}-{month:02d}] "
                    f"No successful days to save.",
                    flush=True
                )

            completed_months += 1

            # Refresh session once per month
            print(
                "Refreshing NASA session...",
                flush=True
            )

            session = make_session(
                get_token()
            )

    # ---------- Final assembly ----------

    print("\n" + "=" * 70, flush=True)

    print(
        "All requested months processed.",
        flush=True
    )

    if missing:

        missing_path = os.path.join(
            OUT_DIR,
            "dust_missing_days.txt"
        )

        with open(
            missing_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(missing)
            )

        print(
            f"{len(missing)} days could not be downloaded.",
            flush=True
        )

        print(
            f"See: {missing_path}",
            flush=True
        )

    # Find cached monthly files
    files = sorted(
        f
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".csv")
    )

    if not files:

        print(
            "No monthly CSV files were created.",
            flush=True
        )

        return

    print(
        f"Combining {len(files)} monthly files...",
        flush=True
    )

    all_df = pd.concat(
        [
            pd.read_csv(
                os.path.join(
                    CACHE_DIR,
                    f
                )
            )
            for f in files
        ],
        ignore_index=True
    )

    all_df = all_df.sort_values(
        "time_utc"
    )

    out_path = os.path.join(
        OUT_DIR,
        "akure_dust_hourly.csv"
    )

    all_df.to_csv(
        out_path,
        index=False
    )

    total_elapsed = (
        time.time()
        - overall_start
    )

    print(
        f"\nDone.",
        flush=True
    )

    print(
        f"Hourly values : {len(all_df):,}",
        flush=True
    )

    print(
        f"Monthly files : {len(files):,}",
        flush=True
    )

    print(
        f"Missing days  : {len(missing):,}",
        flush=True
    )

    print(
        f"Total runtime : {total_elapsed / 3600:.2f} hours",
        flush=True
    )

    print(
        f"Final output  : {out_path}",
        flush=True
    )

    print("=" * 70, flush=True)


# ---------- Main ----------

if __name__ == "__main__":

    if (
        len(sys.argv) > 1
        and sys.argv[1].lower() == "test"
    ):
        test_run()

    else:
        full_run()

