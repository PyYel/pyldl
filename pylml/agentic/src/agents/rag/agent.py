from typing import Any, Optional
from crewai import Agent
from crewai.mcp import MCPServerHTTP 



class RAGAgent(Agent):
    """Overload a CrewAI agent with both custom tools and MCP Server capabilities"""

    def __init__(
        self,
        db_host: str,
        db_user: str,
        db_password: str,
        mcp_urls: list[str] = [],
        mcp_tokens: list[str] = [],
        **kwargs: Any,
    ) -> None:


        # Custom internal tools
        tools = [
        ]

        mcps = []
        for mcp_url, mcp_token in zip(mcp_urls, mcp_tokens):
            hackuity_mcp = MCPServerHTTP(
                url=mcp_url,
                headers={"Authorization": f"Bearer {mcp_token}"},
                cache_tools_list=True,  # Optimization to prevent polling the server every single turn
            )
            mcps.append(hackuity_mcp)

        super().__init__(
            role="RAG browsing agent",
            goal="Find matching sources from a knowledge base.",
            backstory="An expert of <topic>",
            tools=tools,
            mcps=mcps,
            **kwargs,
        )
