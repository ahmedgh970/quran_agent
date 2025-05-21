import os
from dotenv import load_dotenv, find_dotenv
from quran_agent.utils import get_access_token


load_dotenv(find_dotenv(".env", usecwd=True))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in the .env file.")

PREPROD_CLIENT_ID = os.getenv('PREPROD_CLIENT_ID')
PREPROD_CLIENT_SECRET = os.getenv('PREPROD_CLIENT_SECRET')
if not PREPROD_CLIENT_ID or not PREPROD_CLIENT_SECRET:
    raise ValueError("PREPROD_CLIENT_ID and PREPROD_CLIENT_SECRET must be set in the .env file.")

PREPROD_AUTH_URL = "https://prelive-oauth2.quran.foundation/oauth2/token"
PREPROD_BASE_URL = "https://apis-prelive.quran.foundation/content/api/v4"
PREPROD_ACCESS_TOKEN = get_access_token(
    PREPROD_AUTH_URL,
    PREPROD_CLIENT_ID,
    PREPROD_CLIENT_SECRET
)

PROD_CLIENT_ID = os.getenv('PROD_CLIENT_ID')
PROD_CLIENT_SECRET = os.getenv('PROD_CLIENT_SECRET')
if not PROD_CLIENT_ID or not PROD_CLIENT_SECRET:
    raise ValueError("PROD_CLIENT_ID and PROD_CLIENT_SECRET must be set in the .env file.")

PROD_AUTH_URL = "https://oauth2.quran.foundation/oauth2/token"
PROD_BASE_URL = "https://apis.quran.foundation/content/api/v4"
PROD_ACCESS_TOKEN = get_access_token(
    PROD_AUTH_URL, 
    PROD_CLIENT_ID, 
    PROD_CLIENT_SECRET
)
