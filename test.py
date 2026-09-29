import os
import time
import requests
import earthaccess
import xarray as xr


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 1984
MONTH = 12

LAT = 7.15
LON = 5.18

SHORT_NAME = "M2T1NXAER"
VERSION = "5.12.4"

TOKEN_FILE = "earthdata_token.txt"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MERRA-2 MONTHLY TEST")
print("=" * 70)

print(f"Month       : {YEAR}-{MONTH:02d}")
print(f"Akure point : {LAT}, {LON}")
print("Variables   : DUSMASS, DUEXTTAU")


# ============================================================
# LOAD TOKEN
# ============================================================

if not os.path.exists(TOKEN_FILE):
    raise FileNotFoundError(
        f"Could not find {TOKEN_FILE}"
    )

with open(TOKEN_FILE, "r", encoding="utf-8") as f:
    TOKEN = f.read().strip()

if not TOKEN:
    raise ValueError("Earthdata token file is empty.")

# Make the token available to earthaccess.
os.environ["EARTHDATA_TOKEN"] = TOKEN

print("\nEarthdata token loaded.")
print(
    "EARTHDATA_TOKEN present:",
    bool(os.environ.get("EARTHDATA_TOKEN"))
)


# ============================================================
# EARTHDATA CATALOGUE LOGIN
# ============================================================

print("\nLogging into NASA Earthdata...")

auth = earthaccess.login(strategy="environment")

print("Catalogue login successful.")


# ============================================================
# MANUALLY AUTHENTICATED HTTP SESSION
# ============================================================

print("\nCreating authenticated HTTP session...")

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {TOKEN}",
    "User-Agent": "MERRA-2-Akure-Research/1.0"
})

print(
    "Authorization header present:",
    session.headers.get("Authorization", "").startswith("Bearer ")
)


# ============================================================
# SEARCH NASA CATALOGUE
# ============================================================

print("\nSearching NASA catalogue...")

start = time.time()

results = earthaccess.search_data(
    short_name=SHORT_NAME,
    version=VERSION,
    temporal=(
        f"{YEAR}-{MONTH:02d}-01",
        f"{YEAR}-{MONTH:02d}-31"
    ),
)

print(
    f"Found {len(results)} file(s) "
    f"in {time.time() - start:.2f}s"
)

if not results:
    raise RuntimeError("No MERRA-2 files found.")


# ============================================================
# GET FIRST FILE URL
# ============================================================

first_result = results[0]

url = first_result["umm"]["RelatedUrls"][0]["URL"]

print("\nOpening first MERRA-2 dataset through OPeNDAP...")
print("URL:", url)


# ============================================================
# DAP4 CONSTRAINT
# ============================================================

constraint = (
    "dap4.ce="
    "/time,"
    "/lat,"
    "/lon,"
    "/DUSMASS,"
    "/DUEXTTAU"
)

dap_url = (
    "dap4://"
    + url.replace("https://", "")
    + "?"
    + constraint
)

print("\nOpening dataset...")

start = time.time()

ds = xr.open_dataset(
    dap_url,
    engine="pydap",
    session=session
)

print(
    f"Dataset opened in {time.time() - start:.2f}s"
)


# ============================================================
# SHOW DATASET
# ============================================================

print("\nDataset dimensions:")
print(ds.dims)

print("\nVariables:")
print(list(ds.data_vars))


# ============================================================
# SELECT AKURE GRID CELL
# ============================================================

print("\nSelecting nearest Akure grid cell...")

point = ds.sel(
    lat=LAT,
    lon=LON,
    method="nearest"
)

print(
    "Requested:",
    LAT,
    LON
)

print(
    "MERRA-2 grid:",
    float(point.lat.values),
    float(point.lon.values)
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading DUSMASS and DUEXTTAU...")

start = time.time()

data = point[
    [
        "DUSMASS",
        "DUEXTTAU"
    ]
].load()

print(
    f"Data loaded in {time.time() - start:.2f}s"
)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

df = data.to_dataframe().reset_index()

df["dust_ugm3"] = df["DUSMASS"] * 1e9
df["dust_aod"] = df["DUEXTTAU"]

df = df[
    [
        "time",
        "dust_ugm3",
        "dust_aod"
    ]
]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print("Rows:", len(df))
print("First:", df["time"].min())
print("Last :", df["time"].max())

print(
    "Missing dust:",
    df["dust_ugm3"].isna().sum()
)

print(
    "Missing AOD :",
    df["dust_aod"].isna().sum()
)

print("\nSample:")
print(df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)