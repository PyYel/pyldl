from tools import (
    HackuityGetFindingsForRGTool,
    HackuityInterface,
    HackuityGetFindingsTool,
)

import os
from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
    override=False,
)


HACKUITY_API_HOST = os.getenv("HACKUITY_API_HOST", "")
HACKUITY_API_TOKEN = os.getenv("HACKUITY_API_TOKEN", "")
HACKUITY_CORP_ID = os.getenv("HACKUITY_CORP_ID", "")


hackuity_interface = HackuityInterface(
    api_host=HACKUITY_API_HOST,
    access_token=HACKUITY_API_TOKEN,
    corp_id=HACKUITY_CORP_ID,
)


tool = HackuityGetFindingsTool(hackuity_interface=hackuity_interface)
response = tool._run()

print(response)
