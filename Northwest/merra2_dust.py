"""
STEP 1: Download MERRA-2 hourly dust for the grid cell nearest Akure
--------------------------------------------------------------------
Dataset : MERRA-2 M2T1NXAER (hourly aerosol diagnostics), from NASA GES DISC
Variables: DUSMASS (dust surface mass concentration, kg/m3) and
           DUEXTTAU (dust extinction optical thickness at 550 nm)
Output  : outputs/akure_dust_hourly.csv  (hourly values, UTC time)

ONE-TIME SETUP (needed before the first run)
1. Make a free NASA Earthdata account: https://urs.earthdata.nasa.gov
2. Log in to Earthdata, go to Applications > Authorized Apps > Approve More
   Applications, and approve "NASA GESDISC DATA ARCHIVE".
3. Create a file called .netrc (on Windows: _netrc) in your home folder with:
       machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
4. Create a file called .dodsrc in the same home folder with two lines:
       HTTP.COOKIEJAR=C:\\Users\\YOURNAME\\.urs_cookies
       HTTP.NETRC=C:\\Users\\YOURNAME\\_netrc
   (Use your own home folder path. On Linux or Mac use /home/you/... paths.)
5. pip install xarray netCDF4 pandas

FIRST RUN A TEST:   python 01_download_merra2_dust.py test
If the test prints dust values, run the full download:
                    python 01_download_merra2_dust.py

The full run opens about 6,400 daily files, so it can take a few hours.
It saves one small file per month, so you can stop it and run it again
and it will carry on from where it stopped.
"""

import os
import sys
import time
import calendar
import datetime as dt
import pandas as pd
import xarray as xr

# ---------------- Settings ----------------
LAT, LON = 7.15, 5.18            # Akure station coordinates
FIRST_SEASON = 1984              # season 1984 = Nov 1984 to Mar 1985
LAST_SEASON = 2025               # season 2025 = Nov 2025 to Mar 2026
SEASON_MONTHS = [11, 12, 1, 2, 3]
VARS = ["DUSMASS", "DUEXTTAU"]
OUT_DIR = "outputs"
CACHE_DIR = os.path.join(OUT_DIR, "dust_cache")
BASE = "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXAER.5.12.4"
RETRIES = 3
# ------------------------------------------

os.makedirs(CACHE_DIR, exist_ok=True)


def expected_stream(year):
    """MERRA-2 file stream number depends on the year."""
    if year <= 1991:
        return 100
    if year <= 2000:
        return 200
    if year <= 2010:
        return 300
    return 400


def day_url(date, stream):
    return (f"{BASE}/{date.year}/{date.month:02d}/"
            f"MERRA2_{stream}.tavg1_2d_aer_Nx.{date:%Y%m%d}.nc4")


def fetch_day(date):
    """Return (dataframe, None) on success or (None, error) on failure."""
    first = expected_stream(date.year)
    # A few reprocessed months use stream 401, so we try other streams too
    streams = [first] + [s for s in (401, 400, 300, 200, 100) if s != first]
    last_err = None
    for stream in streams:
        url = day_url(date, stream)
        for attempt in range(RETRIES):
            try:
                with xr.open_dataset(url) as ds:
                    i = abs(ds["lat"].values - LAT).argmin()
                    j = abs(ds["lon"].values - LON).argmin()
                    grid_lat = float(ds["lat"].values[i])
                    grid_lon = float(ds["lon"].values[j])
                    sub = ds[VARS].isel(lat=int(i), lon=int(j)).load()
                df = sub.to_dataframe().reset_index()
                out = pd.DataFrame({
                    "time_utc": pd.to_datetime(df["time"].astype(str)),
                    "dust_ugm3": df["DUSMASS"] * 1e9,   # kg/m3 to ug/m3
                    "dust_aod": df["DUEXTTAU"],
                })
                out["grid_lat"] = grid_lat
                out["grid_lon"] = grid_lon
                return out, None
            except Exception as e:  # noqa
                last_err = e
                msg = str(e).lower()
                if "404" in msg or "not found" in msg or "no such file" in msg:
                    break  # wrong stream number, try the next one
                time.sleep(2 * (attempt + 1))
    return None, last_err


def test_run():
    date = dt.date(2019, 12, 15)
    print(f"Test: fetching {date} ...")
    df, err = fetch_day(date)
    if df is None:
        print("FAILED. Last error:", err)
        print("Check the setup steps at the top of this file "
              "(.netrc, .dodsrc, and the GESDISC app approval).")
        return
    print(df.head(24).to_string(index=False))
    print(f"\nGrid cell used: lat {df['grid_lat'][0]}, lon {df['grid_lon'][0]}")
    print("Test passed. Now run without 'test' for the full download.")


def full_run():
    latest_ok = dt.date.today() - dt.timedelta(days=35)   # MERRA-2 has a lag
    missing = []
    fails_in_a_row = 0
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        for month in SEASON_MONTHS:
            year = season if month >= 11 else season + 1
            cache = os.path.join(CACHE_DIR, f"dust_{year}_{month:02d}.csv")
            if os.path.exists(cache):
                continue
            frames = []
            ndays = calendar.monthrange(year, month)[1]
            for d in range(1, ndays + 1):
                date = dt.date(year, month, d)
                if date > latest_ok:
                    continue
                df, err = fetch_day(date)
                if df is None:
                    missing.append(str(date))
                    fails_in_a_row += 1
                    if fails_in_a_row >= 5 and not frames:
                        print("Five failures in a row. Last error:", err)
                        print("Stopping. Check your setup (see top of file).")
                        return
                else:
                    fails_in_a_row = 0
                    frames.append(df)
            if frames:
                pd.concat(frames).to_csv(cache, index=False)
            print(f"Saved {year}-{month:02d} ({len(frames)} of {ndays} days)")

    if missing:
        with open(os.path.join(OUT_DIR, "dust_missing_days.txt"), "w") as f:
            f.write("\n".join(missing))
        print(f"{len(missing)} days could not be downloaded (see dust_missing_days.txt)")

    files = sorted(os.listdir(CACHE_DIR))
    all_df = pd.concat(
        [pd.read_csv(os.path.join(CACHE_DIR, f)) for f in files if f.endswith(".csv")]
    ).sort_values("time_utc")
    out_path = os.path.join(OUT_DIR, "akure_dust_hourly.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Done. Saved {len(all_df)} hourly values to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        test_run()
    else:
        full_run()
