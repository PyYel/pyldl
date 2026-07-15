from typing import Any, Optional
from crewai import Agent
from crewai.mcp import MCPServerHTTP 

from tools import HackuityGetAssetsTool, HackuityGetFindingsTool
from hackuity import HackuityInterface


class HackuityAgent(Agent):
    """Overload a CrewAI agent with both custom tools and MCP Server capabilities"""

    def __init__(
        self,
        api_host: str,
        access_token: str,
        corp_id: str,
        mcp_urls: list[str] = [],
        **kwargs: Any,
    ) -> None:

        self.client = HackuityInterface(
            api_host=api_host, access_token=access_token, corp_id=corp_id
        )

        # Custom internal tools
        tools = [
            HackuityGetAssetsTool(hackuity_interface=self.client),
            HackuityGetFindingsTool(hackuity_interface=self.client),
        ]

        mcps = []
        for mcp_url in mcp_urls:
            hackuity_mcp = MCPServerHTTP(
                url=mcp_url,
                headers={"Authorization": f"Bearer {access_token}"},
                cache_tools_list=True,  # Optimization to prevent polling the server every single turn
            )
            mcps.append(hackuity_mcp)

        super().__init__(
            role="Hackuity SecOps Engineer",
            goal="Analyze infrastructure assets and active threat vulnerabilities.",
            backstory="An automated incident responder specialized in vulnerability platforms.",
            tools=tools,
            mcps=mcps,
            **kwargs,
        )
