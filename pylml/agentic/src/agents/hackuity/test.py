

from agent import HackuityAgent

from dotenv import load_dotenv
import os

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
    override=False,
)


HACKUITY_API_HOST = os.getenv("HACKUITY_API_HOST", "")
HACKUITY_API_TOKEN = os.getenv("HACKUITY_API_TOKEN", "")
HACKUITY_CORP_ID = os.getenv("HACKUITY_CORP_ID", "")


agent = HackuityAgent(
    api_host=HACKUITY_API_HOST,
    access_token=HACKUITY_API_TOKEN,
    corp_id=HACKUITY_CORP_ID,
)

