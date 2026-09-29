"""
MERRA-2 hourly dust downloader for Akure, Nigeria.

Dataset:
    MERRA-2 M2T1NXAER v5.12.4

Variables:
    DUSMASS  - dust surface mass concentration (kg/m3)
    DUEXTTAU - dust extinction optical thickness at 550 nm

Target:
    Akure, nearest MERRA-2 grid cell to:
        LAT = 7.15
        LON = 5.18

Storage architecture:
    NASA GES DISC
        ↓
    Railway temporary filesystem
        ↓
    Monthly CSV
        ↓
    Google Drive
        ↓
    Temporary local CSV deleted

Google Drive is the persistent checkpoint.

Run locally:
    python 01_download_merra2_dust.py test

Run full download:
    python 01_download_merra2_dust.py

Environment variables required:
    EARTHDATA_TOKEN

    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    GOOGLE_REFRESH_TOKEN
    GOOGLE_DRIVE_FOLDER_ID

Optional:
    EARTHDATA_USER_AGENT
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

# Load local .env if present (optional; avoids manual env setup)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# ============================================================
# SETTINGS
# ============================================================

LAT = 7.15
LON = 5.18

FIRST_SEASON = 1984
LAST_SEASON = 2025

SEASON_MONTHS = [11, 12, 1, 2, 3]

VARS = [
    "DUSMASS",
    "DUEXTTAU",
]

SHORT_NAME = "M2T1NXAER"
VERSION = "5.12.4"

# Temporary local working directory.
OUT_DIR = "outputs"
CACHE_DIR = os.path.join(OUT_DIR, "dust_cache")

RETRIES = 3

DRIVE_FOLDER_ID = os.environ.get(
    "GOOGLE_DRIVE_FOLDER_ID",
    ""
).strip()

EARTHDATA_TOKEN = os.environ.get(
    "EARTHDATA_TOKEN",
    ""
).strip()

USER_AGENT = os.environ.get(
    "EARTHDATA_USER_AGENT",
    "Akure-MERRA2-Dust-Research/1.0"
)


# ============================================================
# LOCAL DIRECTORIES
# ============================================================

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# DAP4 CONSTRAINT
# ============================================================

CONSTRAINT = (
    "?dap4.ce="
    + ";".join(
        f"/{v}"
        for v in VARS + ["lat", "lon", "time"]
    ).replace(";", "%3B")
)


# ============================================================
# ENVIRONMENT CHECK
# ============================================================

def validate_environment():
    """Check that required Railway/local environment variables exist."""

    missing = []

    if not EARTHDATA_TOKEN:
        missing.append("EARTHDATA_TOKEN")

    if not os.environ.get("GOOGLE_CLIENT_ID", "").strip():
        missing.append("GOOGLE_CLIENT_ID")

    if not os.environ.get("GOOGLE_CLIENT_SECRET", "").strip():
        missing.append("GOOGLE_CLIENT_SECRET")

    if not os.environ.get("GOOGLE_REFRESH_TOKEN", "").strip():
        missing.append("GOOGLE_REFRESH_TOKEN")

    if not DRIVE_FOLDER_ID:
        missing.append("GOOGLE_DRIVE_FOLDER_ID")

    if missing:
        raise RuntimeError(
            "Missing required environment variables:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )


# ============================================================
# NASA SESSION
# ============================================================

def make_session(token):
    """
    Create an authenticated HTTP session for NASA GES DISC.

    PyDAP's create_session is preferred.
    Requests is used as fallback.
    """

    try:
        from pydap.net import create_session

        return create_session(
            session_kwargs={
                "token": token,
                "headers": {
                    "User-Agent": USER_AGENT
                }
            }
        )

    except Exception:

        import requests

        # Prefer earthaccess-based authenticated cookies when available.
        try:
            import earthaccess

            auth = earthaccess.login(strategy="environment")

            session = requests.Session()

            # Try common attribute names where earthaccess may store a requests.Session
            cookie_source = None

            for attr in ("session", "_session", "requests_session", "client"):
                if hasattr(auth, attr):
                    cookie_source = getattr(auth, attr)
                    break

            # If the auth object itself exposes a cookie jar, use it
            if cookie_source is None and hasattr(auth, "cookie_jar"):
                cookie_source = auth

            # Transfer cookies if possible
            try:
                cookies = getattr(cookie_source, "cookies", None)

                if cookies is not None:
                    session.cookies.update(cookies)

            except Exception:
                # best-effort; ignore failures and fall back
                pass

            session.headers.update({
                "User-Agent": USER_AGENT
            })

            return session

        except Exception:

            session = requests.Session()

            session.headers.update({
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
            })

            return session



# ============================================================
# NASA GRANULE DISCOVERY
# ============================================================

def opendap_url(granule):
    """Build the Cloud OPeNDAP URL for one daily MERRA-2 file."""

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

    cid = umm["meta"]["collection-concept-id"]
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

    if not m:
        return None

    return dt.datetime.strptime(
        m.group(1),
        "%Y%m%d"
    ).date()


def find_granules(start, end):

    import earthaccess

    results = earthaccess.search_data(
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

    output = {}

    for granule in results:

        d = granule_date(granule)

        if d is not None and start <= d <= end:
            output[d] = granule

    return output


# ============================================================
# DAILY EXTRACTION
# ============================================================

def extract_point(ds):
    """
    Extract the MERRA-2 grid cell nearest Akure.
    """

    lat_values = ds["lat"].values
    lon_values = ds["lon"].values

    i = int(
        abs(lat_values - LAT).argmin()
    )

    j = int(
        abs(lon_values - LON).argmin()
    )

    grid_lat = float(lat_values[i])
    grid_lon = float(lon_values[j])

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

    output = pd.DataFrame({
        "time_utc": pd.to_datetime(
            df["time"].astype(str)
        ),

        "dust_ugm3": (
            df["DUSMASS"] * 1e9
        ),

        "dust_aod": df["DUEXTTAU"],
    })

    output["grid_lat"] = grid_lat
    output["grid_lon"] = grid_lon

    return output


def fetch_day(granule, session):

    url = opendap_url(granule) + CONSTRAINT

    last_error = None

    for attempt in range(RETRIES):

        try:

            # pydap/xarray prefers the dap4:// scheme for OPeNDAP access
            dap_url = url

            if dap_url.startswith("https://"):
                dap_url = dap_url.replace("https://", "dap4://", 1)

            # Ensure Authorization header is present when we have a token
            try:
                session.headers.update({
                    "Authorization": f"Bearer {EARTHDATA_TOKEN}"
                })
            except Exception:
                pass

            with xr.open_dataset(
                dap_url,
                engine="pydap",
                session=session
            ) as ds:

                return extract_point(ds), None

        except Exception as error:

            last_error = error

            if attempt < RETRIES - 1:

                wait = 2 * (attempt + 1)

                print(
                    f" retry {attempt + 1}/{RETRIES - 1}"
                    f" in {wait}s...",
                    end="",
                    flush=True
                )

                time.sleep(wait)

    return None, last_error


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    """
    Build an authenticated Google Drive API client
    using an OAuth refresh token.

    No browser/login is required on Railway.
    """

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    client_id = os.environ[
        "GOOGLE_CLIENT_ID"
    ]

    client_secret = os.environ[
        "GOOGLE_CLIENT_SECRET"
    ]

    refresh_token = os.environ[
        "GOOGLE_REFRESH_TOKEN"
    ]

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=[
            "https://www.googleapis.com/auth/drive"
        ],
    )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def drive_file_exists(service, filename):
    """
    Check whether a completed monthly CSV already exists
    in the configured Google Drive folder.
    """

    query = (
        f"name = '{filename.replace(chr(39), chr(92) + chr(39))}' "
        f"and '{DRIVE_FOLDER_ID}' in parents "
        f"and trashed = false"
    )

    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="files(id,name,size,modifiedTime)",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            pageSize=10,
        )
        .execute()
    )

    files = response.get("files", [])

    return files[0] if files else None


def upload_to_drive(service, local_path, filename):
    """
    Upload a monthly CSV to Google Drive.

    Resumable upload is used for reliability.
    """

    from googleapiclient.http import MediaFileUpload

    existing = drive_file_exists(
        service,
        filename
    )

    media = MediaFileUpload(
        local_path,
        mimetype="text/csv",
        resumable=True,
        chunksize=1024 * 1024,
    )

    if existing:

        print(
            f"Google Drive already contains "
            f"{filename}. Skipping upload.",
            flush=True
        )

        return existing["id"]

    metadata = {
        "name": filename,
        "parents": [DRIVE_FOLDER_ID],
    }

    request = (
        service.files()
        .create(
            body=metadata,
            media_body=media,
            fields="id,name,size",
            supportsAllDrives=True,
        )
    )

    response = None

    while response is None:

        status, response = request.next_chunk()

        if status:

            print(
                f"  Drive upload: "
                f"{int(status.progress() * 100)}%",
                flush=True
            )

    print(
        f"  Uploaded to Drive: "
        f"{response['name']} "
        f"({response.get('size', '?')} bytes)",
        flush=True
    )

    return response["id"]


# ============================================================
# TEST MODE
# ============================================================

def test_run():

    print("=" * 70)
    print("MERRA-2 / GOOGLE DRIVE TEST")
    print("=" * 70)

    validate_environment()

    day = dt.date(2019, 12, 15)

    print(
        f"Test date: {day}",
        flush=True
    )

    print(
        "Searching NASA catalogue...",
        flush=True
    )

    granules = find_granules(
        day,
        day
    )

    if day not in granules:

        raise RuntimeError(
            "Could not find the requested MERRA-2 file."
        )

    granule = granules[day]

    print(
        "OPeNDAP address:",
        opendap_url(granule),
        flush=True
    )

    session = make_session(
        EARTHDATA_TOKEN
    )

    print(
        "Requesting Akure grid cell...",
        flush=True
    )

    start_time = time.time()

    df, error = fetch_day(
        granule,
        session
    )

    elapsed = (
        time.time()
        - start_time
    )

    if df is None:

        raise RuntimeError(
            f"NASA request failed after "
            f"{elapsed:.1f}s:\n{error}"
        )

    print(
        f"NASA extraction successful "
        f"({elapsed:.1f}s)",
        flush=True
    )

    print(
        df.head(24).to_string(index=False),
        flush=True
    )

    print(
        f"\nGrid cell:"
        f" {df['grid_lat'].iloc[0]} N,"
        f" {df['grid_lon'].iloc[0]} E",
        flush=True
    )

    # --------------------------------------------------------
    # Test Drive authentication too
    # --------------------------------------------------------

    print(
        "\nTesting Google Drive...",
        flush=True
    )

    drive = get_drive_service()

    about = (
        drive.about()
        .get(fields="user(displayName,emailAddress)")
        .execute()
    )

    print(
        "Google account:",
        about["user"]["emailAddress"],
        flush=True
    )

    folder = (
        drive.files()
        .get(
            fileId=DRIVE_FOLDER_ID,
            fields="id,name,mimeType",
            supportsAllDrives=True,
        )
        .execute()
    )

    print(
        "Drive folder:",
        folder["name"],
        flush=True
    )

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)

    print(
        "NASA authentication: OK",
        flush=True
    )

    print(
        "MERRA-2 extraction: OK",
        flush=True
    )

    print(
        "Google Drive authentication: OK",
        flush=True
    )

    print(
        "Google Drive folder access: OK",
        flush=True
    )


# ============================================================
# FULL DOWNLOAD
# ============================================================

def full_run():

    validate_environment()

    latest_allowed = (
        dt.date.today()
        - dt.timedelta(days=35)
    )

    drive = get_drive_service()

    session = make_session(
        EARTHDATA_TOKEN
    )

    total_months = (
        (LAST_SEASON - FIRST_SEASON + 1)
        * len(SEASON_MONTHS)
    )

    completed_months = 0
    total_days_downloaded = 0

    missing_days = []

    print("=" * 70)
    print("MERRA-2 AKURE DUST DOWNLOAD")
    print("=" * 70)

    print(
        f"Seasons       : "
        f"{FIRST_SEASON}-{LAST_SEASON}"
    )

    print(
        f"Months/season : "
        f"{', '.join(map(str, SEASON_MONTHS))}"
    )

    print(
        f"Akure target  : "
        f"{LAT}, {LON}"
    )

    print(
        f"Drive folder  : "
        f"{DRIVE_FOLDER_ID}"
    )

    print(
        f"Latest allowed: "
        f"{latest_allowed}"
    )

    print("=" * 70)

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

            filename = (
                f"dust_{year}_{month:02d}.csv"
            )

            # ------------------------------------------------
            # CHECK GOOGLE DRIVE FIRST
            # ------------------------------------------------

            print(
                f"\n[{year}-{month:02d}] "
                f"Checking Google Drive...",
                flush=True
            )

            try:

                existing = drive_file_exists(
                    drive,
                    filename
                )

            except Exception as error:

                print(
                    f"[{year}-{month:02d}] "
                    f"Drive check failed: {error}",
                    flush=True
                )

                raise

            if existing:

                print(
                    f"[{year}-{month:02d}] "
                    f"Already on Google Drive — "
                    f"skipping.",
                    flush=True
                )

                completed_months += 1

                continue

            # ------------------------------------------------
            # DATE RANGE
            # ------------------------------------------------

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
                latest_allowed
            )

            if end < start:

                print(
                    f"[{year}-{month:02d}] "
                    f"Future/unavailable — skipping.",
                    flush=True
                )

                continue

            print(
                "\n" + "=" * 70,
                flush=True
            )

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

            except Exception as error:

                print(
                    f"[{year}-{month:02d}] "
                    f"CATALOGUE SEARCH FAILED:",
                    error,
                    flush=True
                )

                continue

            search_elapsed = (
                time.time()
                - search_start
            )

            print(
                f"[{year}-{month:02d}] "
                f"Found {len(granules)} "
                f"daily files "
                f"in {search_elapsed:.1f}s",
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

            failures = 0

            days = sorted(
                granules
            )

            month_start = time.time()

            for n, day in enumerate(
                days,
                1
            ):

                print(
                    f"[{year}-{month:02d}] "
                    f"Day {n}/{len(days)} "
                    f"({day}) ...",
                    end=" ",
                    flush=True
                )

                day_start = time.time()

                df, error = fetch_day(
                    granules[day],
                    session
                )

                elapsed = (
                    time.time()
                    - day_start
                )

                if df is None:

                    print(
                        f"FAILED after "
                        f"{elapsed:.1f}s",
                        flush=True
                    )

                    missing_days.append(
                        str(day)
                    )

                    failures += 1

                    # Stop only if NASA appears to have
                    # become completely unavailable.
                    if (
                        failures >= 5
                        and not frames
                    ):

                        print(
                            "\nFive consecutive "
                            "failures with no "
                            "successful downloads.",
                            flush=True
                        )

                        print(
                            "Last error:",
                            error,
                            flush=True
                        )

                        raise RuntimeError(
                            "NASA data requests "
                            "appear unavailable."
                        )

                else:

                    frames.append(df)

                    total_days_downloaded += 1

                    failures = 0

                    print(
                        f"OK ({elapsed:.1f}s)",
                        flush=True
                    )

            # ------------------------------------------------
            # SAVE AND UPLOAD MONTH
            # ------------------------------------------------

            if not frames:

                print(
                    f"[{year}-{month:02d}] "
                    f"No successful days. "
                    f"Nothing uploaded.",
                    flush=True
                )

                continue

            month_df = pd.concat(
                frames,
                ignore_index=True
            )

            local_path = os.path.join(
                CACHE_DIR,
                filename
            )

            month_df.to_csv(
                local_path,
                index=False
            )

            month_elapsed = (
                time.time()
                - month_start
            )

            print(
                f"[{year}-{month:02d}] "
                f"Created local CSV: "
                f"{len(month_df)} rows",
                flush=True
            )

            print(
                f"[{year}-{month:02d}] "
                f"Month processing time: "
                f"{month_elapsed / 60:.1f} minutes",
                flush=True
            )

            # ------------------------------------------------
            # UPLOAD TO DRIVE
            # ------------------------------------------------

            print(
                f"[{year}-{month:02d}] "
                f"Uploading to Google Drive...",
                flush=True
            )

            upload_to_drive(
                drive,
                local_path,
                filename
            )

            # ------------------------------------------------
            # DELETE TEMPORARY LOCAL FILE
            # ------------------------------------------------

            try:

                os.remove(
                    local_path
                )

                print(
                    f"[{year}-{month:02d}] "
                    f"Temporary local file deleted.",
                    flush=True
                )

            except OSError as error:

                print(
                    f"Warning: could not delete "
                    f"{local_path}: {error}",
                    flush=True
                )

            completed_months += 1

            print(
                f"[{year}-{month:02d}] "
                f"COMPLETE — "
                f"{completed_months}/{total_months} "
                f"months processed.",
                flush=True
            )

            # Refresh NASA session once per month.
            print(
                "Refreshing NASA session...",
                flush=True
            )

            session = make_session(
                EARTHDATA_TOKEN
            )

    # ========================================================
    # FINISHED
    # ========================================================

    total_elapsed = (
        time.time()
        - overall_start
    )

    print(
        "\n" + "=" * 70,
        flush=True
    )

    print(
        "DOWNLOAD RUN FINISHED",
        flush=True
    )

    print(
        "=" * 70,
        flush=True
    )

    print(
        f"Months completed : "
        f"{completed_months}/{total_months}",
        flush=True
    )

    print(
        f"Days downloaded  : "
        f"{total_days_downloaded}",
        flush=True
    )

    print(
        f"Missing days     : "
        f"{len(missing_days)}",
        flush=True
    )

    print(
        f"Runtime          : "
        f"{total_elapsed / 3600:.2f} hours",
        flush=True
    )

    if missing_days:

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
                "\n".join(missing_days)
            )

        print(
            f"Missing-day list : "
            f"{missing_path}",
            flush=True
        )

    print(
        "\nMonthly CSVs are stored in Google Drive.",
        flush=True
    )

    print(
        "The monthly files can be combined later "
        "into the final Akure dataset.",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if (
        len(sys.argv) > 1
        and sys.argv[1].lower() == "test"
    ):
        test_run()

    else:
        full_run()