import asyncio
import os
import sys
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
import httpx

MODULE_DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if __name__ == "__main__":
    sys.path.append(MODULE_DIR_PATH)

load_dotenv(
    os.path.join(MODULE_DIR_PATH, ".env"),
    override=False,
)


MCP_TOKEN = os.getenv("MCP_TOKEN", "")
MCP_HOST = os.getenv("MCP_HOST", "")

async def fetch_tools():
    headers = {
        "Authorization": f"Bearer {MCP_TOKEN}",
    }

    print(f"Connecting to Streamable HTTP server at {MCP_HOST}...")

    async with httpx.AsyncClient(headers=headers) as http_client:
        try:
            async with streamable_http_client(MCP_HOST, http_client=http_client) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    print("Initializing session...")
                    await session.initialize()

                    tools_response = await session.list_tools()
                    print("\n--- Available Tools ---")
                    for tool in tools_response.tools:
                        print(f"- {tool.name}: {tool.description}")

        except BaseException as e:
            print("\n--- Connection Failure Details ---")
            if isinstance(e, ExceptionGroup):
                for sub_exc in e.exceptions:
                    print(f"Sub-Exception: {type(sub_exc).__name__} - {sub_exc}")
            else:
                print(f"Exception: {type(e).__name__} - {e}")


asyncio.run(fetch_tools())
