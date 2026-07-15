from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
from pydantic import BaseModel, Field, AliasChoices, AliasPath

from crewai.tools import BaseTool
from hackuity.openapi_client.models.api_search_asset import ApiSearchAsset
from hackuity.openapi_client.models.api_search_finding import ApiSearchFinding

from hackuity import HackuityInterface


class Asset(BaseModel):
    """
    Reduced asset schema containing only the critical information
    needed by an LLM, preventing context window bloat.
    """

    id: str = Field(..., description="The unique identifier of the asset.")
    name: str = Field(..., description="The asset name or hostname.")
    ip_addresses: List[str] = Field(
        default_factory=list, description="Assigned network interfaces."
    )
    perimeter_ids: List[str] = Field(
        default_factory=list, description="IDs of matching organizational scopes."
    )
    criticality: Optional[int] = Field(
        None, description="The numerical criticality risk level of this asset."
    )

    # Placeholder for customized parsing of the heavy, nested asset details dictionary
    details_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Truncated and key-value filtered subset of the raw asset details dictionary.",
    )


class Finding(BaseModel):
    """
    A simplified vulnerability finding schema optimized for LLM decision making. You should only add to this model what is relevant to
    an LLm, agent, or user chatting.

    Devlopper note:
    - The hugly Field model definition is meant to be expanded to match the Hackuity JSON response format.
    - If the API response change, just edit this BaseModel to match the python key to the hackuity internal naming
    - This means the tool code will not have to be edited ever using the ``**Class().model_dump()`` trick :D
    """

    id: str = Field(
        ...,
        description="The unique database identifier.",
        validation_alias=AliasChoices("id", "finding_id", "findingId"),
    )
    title: str = Field(
        "Untitled Finding",
        description="The vulnerability signature name.",
        validation_alias=AliasChoices(
            "title",
            "finding_title",
            AliasPath("assessments_related", "by_extended_attributes", 0, "title"),
            AliasPath("by_extended_attributes_titles", 0),
        ),
    )
    severity: str = Field(
        "UNKNOWN",
        description="The risk rating.",
        validation_alias=AliasChoices("severity", "finding_severity"),
    )
    cvss_score: Optional[float] = Field(
        None,
        description="The numerical CVSS rating.",
        validation_alias=AliasChoices(
            "cvss_score", "cvssScore", AliasPath("finding_status", "score", "base")
        ),
    )
    asset_id: Optional[str] = Field(
        None,
        description="The impacted asset ID.",
        validation_alias=AliasChoices("asset_id", "assetId"),
    )
    remediation_status: Optional[str] = Field(
        "OPEN",
        description="Current resolution state.",
        validation_alias=AliasChoices(
            "remediation_status",
            "remediationStatus",
            "status",
            AliasPath("finding_status", "state"),
        ),
    )


class AssetsInput(BaseModel):
    """Input argument validation schema for HackuityGetAssetsTool."""

    page_size: int = Field(
        default=500,
        description="The paginated query chunk limit (max 500 recommended per cycle).",
    )


class FindingsInput(BaseModel):
    """Input argument validation schema for HackuityGetFindingsTool."""

    page_size: int = Field(
        default=500,
        description="The paginated query chunk limit (max 500 recommended per cycle).",
    )


class FindingsForRGInput(BaseModel):
    """Input argument validation schema for HackuityGetFindingsForRGTool."""

    rg_ids: List[str] = Field(
        ...,
        description="A list of specific Remediation Group UUID strings to isolate findings for.",
    )
    page_size: int = Field(
        default=500,
        description="The paginated query chunk limit (max 500 recommended per cycle).",
    )


class HackuityGetAssetsTool(BaseTool):
    name: str = "get_hackuity_assets"
    description: str = (
        "Returns a cleaned, lightweight list of corporate assets tracked inside Hackuity, "
        "stripping heavy metadata payloads to prevent LLM context-window saturation."
    )
    args_schema: type[BaseModel] = AssetsInput
    hackuity_interface: HackuityInterface

    # Required: allows arbitrary non-pydantic types (such as HackuityInterface) on the BaseTool class
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, page_size: int = 500) -> List[Dict[str, Any]]:
        self.hackuity_interface.logger.info(
            "CrewAI executing HackuityGetAssetsTool (page_size: %d)", page_size
        )
        raw_assets = self.hackuity_interface.get_assets(page_size=page_size)
        return parse_and_reduce_assets(raw_assets)


class HackuityGetFindingsTool(BaseTool):
    name: str = "get_all_hackuity_findings"
    description: str = (
        "Retrieves a cleaned and filtered list of active vulnerability findings across the organization. "
        "Excludes verbose trace logs, scanner payloads, and nested remediation details."
    )
    args_schema: type[BaseModel] = FindingsInput
    hackuity_interface: HackuityInterface

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, page_size: int = 500) -> List[Dict[str, Any]]:
        self.hackuity_interface.logger.info(
            "CrewAI executing HackuityGetFindingsTool (page_size: %d)", page_size
        )
        raw_findings = self.hackuity_interface.get_all_findings(page_size=page_size)

        print(len(raw_findings))

        findings = [Finding(**finding.to_dict()) for finding in raw_findings]

        return findings


class HackuityGetFindingsForRGTool(BaseTool):
    name: str = "get_findings_for_remediation_groups"
    description: str = (
        "Gathers simplified, high-priority vulnerability findings restricted to specific "
        "organizational Remediation Group UUIDs."
    )
    args_schema: type[BaseModel] = FindingsForRGInput
    hackuity_interface: HackuityInterface

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, rg_ids: List[str], page_size: int = 500) -> List[Dict[str, Any]]:
        self.hackuity_interface.logger.info(
            "CrewAI executing HackuityGetFindingsForRGTool (RGs: %d, page_size: %d)",
            len(rg_ids),
            page_size,
        )
        raw_findings = self.hackuity_interface.get_findings_for_remediation_groups(
            rg_ids=rg_ids, page_size=page_size
        )
        print(len(raw_findings))

        findings = [Finding(**finding.to_dict()) for finding in raw_findings]

        return findings
