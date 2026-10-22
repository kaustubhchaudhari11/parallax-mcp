"""Factual MCP server for the synthetic incident store.

Run it with ``python -m parallax.server``. The process speaks MCP over
stdio and does not write the observation stream to stdout. Logging from
the SDK is configured onto stderr.

An unknown incident id returns a tool result with ``is_error=True``.
Raising ``ToolError`` would leave ``structured_content`` empty, so the
handler returns ``CallToolResult`` instead. The success annotation is
the observation model, which becomes the advertised output schema. The
SDK skips that schema when ``is_error`` is true.
"""

from typing import cast

from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from parallax.fixtures import IncidentFixture, get_fixture
from parallax.observe import canonical_json
from parallax.schemas import (
    DeploymentInfo,
    IncidentLogs,
    ServiceMetrics,
    UnknownIncidentError,
)

_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def unknown_incident_result(incident_id: str) -> CallToolResult:
    """Build the explicit unknown-incident tool error."""

    payload = UnknownIncidentError(error="unknown_incident", incident_id=incident_id).model_dump(mode="json")
    return CallToolResult(
        content=[TextContent(type="text", text=canonical_json(payload))],
        structured_content=payload,
        is_error=True,
    )


def _resolve(incident_id: str) -> IncidentFixture | CallToolResult:
    fixture = get_fixture(incident_id)
    if fixture is None:
        return unknown_incident_result(incident_id)
    return fixture


def create_server() -> MCPServer:
    """Register the three observation tools on a new server."""

    server = MCPServer(name="parallax", version="0.1.0", log_level="ERROR")

    @server.tool(annotations=_READ_ONLY)
    def get_metrics(incident_id: str) -> ServiceMetrics:
        """Return synthetic service metrics for one incident."""

        found = _resolve(incident_id)
        if isinstance(found, CallToolResult):
            return cast(ServiceMetrics, found)
        return found.metrics.model_copy(deep=True)

    @server.tool(annotations=_READ_ONLY)
    def get_logs(incident_id: str) -> IncidentLogs:
        """Return synthetic log lines for one incident."""

        found = _resolve(incident_id)
        if isinstance(found, CallToolResult):
            return cast(IncidentLogs, found)
        return found.logs.model_copy(deep=True)

    @server.tool(annotations=_READ_ONLY)
    def get_deployment(incident_id: str) -> DeploymentInfo:
        """Return the synthetic deployment record for one incident."""

        found = _resolve(incident_id)
        if isinstance(found, CallToolResult):
            return cast(DeploymentInfo, found)
        return found.deployment.model_copy(deep=True)

    return server


def main() -> None:
    """Serve the three tools on stdio."""

    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
