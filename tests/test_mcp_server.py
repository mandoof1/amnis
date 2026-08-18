"""The MCP surface. In 0.1 this module did not even import."""

from __future__ import annotations

import asyncio


def test_module_imports_and_exposes_tools():
    from amnis.server.mcp import server

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    # 0.1 had `"default": false` (a JS literal) inside a Python dict, so this
    # module raised SyntaxError and every client saw zero tools.
    assert len(tools) >= 20
    for expected in (
        "amnis_remember",
        "amnis_recall",
        "amnis_forget",
        "amnis_search",
        "amnis_compile_wiki",
        "amnis_status",
        "amnis_prune_memory",
        "amnis_update_memory",
    ):
        assert expected in names


def test_schemas_are_valid_json():
    import json

    from amnis.server.mcp import server

    for tool in asyncio.run(server.list_tools()):
        json.dumps(tool.inputSchema)
        assert tool.description, f"{tool.name} has no description"


def test_prune_default_is_a_real_boolean():
    from amnis.server.mcp import server

    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    schema = tools["amnis_prune_memory"].inputSchema
    assert schema["properties"]["dry_run"]["default"] is False


def test_tool_descriptions_are_not_personalised():
    from amnis.server.mcp import server

    joined = " ".join(t.description or "" for t in asyncio.run(server.list_tools()))
    for leaked in (" LO ", "LO's", "himself"):
        assert leaked not in joined
