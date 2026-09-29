# save as check_drive.py and run: python check_drive.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import os

# Load .env for local testing (optional)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

client_id = os.environ.get('GOOGLE_CLIENT_ID')
client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
refresh_token = os.environ.get('GOOGLE_REFRESH_TOKEN')

missing = [
    name for name, val in (
        ('GOOGLE_CLIENT_ID', client_id),
        ('GOOGLE_CLIENT_SECRET', client_secret),
        ('GOOGLE_REFRESH_TOKEN', refresh_token),
    ) if not val
]

if missing:
    raise RuntimeError(
        'Missing required Google environment variables: ' + ', '.join(missing) +
        "\nAdd them to a .env file or export them in your shell."
    )

creds = Credentials(
    token=None,
    refresh_token=refresh_token,
    token_uri='https://oauth2.googleapis.com/token',
    client_id=client_id,
    client_secret=client_secret,
    scopes=['https://www.googleapis.com/auth/drive']
)
drive = build('drive','v3',credentials=creds,cache_discovery=False)

folder_id = '18jhKUXS5kL6j7UCn1zmTcq6VM1ggIZO'  # replace with your ID
meta = drive.files().get(
    fileId=folder_id,
    fields='id,name,owners,permissions,trashed',
    supportsAllDrives=True
).execute()
print(meta)