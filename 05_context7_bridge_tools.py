import asyncio
import copy
import json
from os import environ
import re
import typing

import mcp
from mcp.client import stdio


SERVER_LABEL = "context7"  # prefix to avoid name collisions across servers


def _sanitize_function_name(name: str, prefix: str = SERVER_LABEL) -> str:
    """
    Convert an MCP tool name into a valid OpenAI function name:
      • characters: [a-zA-Z0-9_]
      • starts with a letter/underscore
      • reasonable length (<= 64)
      • prefixed with the server label to avoid collisions
    """
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9_]", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    if not n or not re.match(r"^[a-zA-Z_]", n):
        n = f"tool_{n or 'mcp'}"
    if prefix:
        n = f"{prefix}__{n}"
    return n[:64]


def _normalize_schema(s: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """
    Normalize the MCP tool's inputSchema to a safe JSON Schema for OpenAI function
    'parameters'. We:
      • deep-copy
      • drop '$schema' (noise)
      • ensure 'type' and 'properties' for object schemas
    """
    if not isinstance(s, dict):
        return {"type": "object", "properties": {}}

    out = copy.deepcopy(s)
    out.pop("$schema", None)

    if "type" not in out:
        out["type"] = "object"

    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}

    return out


def bridge_mcp_tools_to_function_tools(
    mcp_tools: list[typing.Any], server_label: str = SERVER_LABEL
) -> tuple[list[dict[str, typing.Any]], dict[str, dict[str, str]]]:
    """
    Convert MCP Tool objects to **Responses API function tools** (flattened).

    Returns:
      (tools, reverse_index)
        tools: list of {
          "type": "function",
          "name": "<server>__<tool>",
          "description": "...",
          "parameters": { ... },
          "strict": true
        }
        reverse_index: function_name -> {"server_label": ..., "mcp_name": ...}
    """
    tools: list[dict[str, typing.Any]] = []
    reverse_index: dict[str, dict[str, str]] = {}

    for t in mcp_tools:
        # robust to snake/camel case
        name = getattr(t, "name", None) or getattr(t, "toolName", None)
        desc = getattr(t, "description", None) or getattr(t, "title", "") or ""
        schema = (
            getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
        )
        if not name:
            continue

        fn_name = _sanitize_function_name(name, prefix=server_label)
        parameters = _normalize_schema(schema)

        tools.append(
            {
                "type": "function",
                "name": fn_name,
                "description": desc or name,
                "parameters": parameters,
            }
        )
        reverse_index[fn_name] = {"server_label": server_label, "mcp_name": name}

    return tools, reverse_index


async def main():
    # Launch the Context7 MCP server over stdio (same as your example)
    stdio_params = mcp.StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@upstash/context7-mcp",
            "--api-key",
            environ.get("CONTEXT7_API_KEY", ""),
        ],
    )

    async with stdio.stdio_client(stdio_params) as (reader, writer):
        async with mcp.ClientSession(reader, writer) as session:
            await session.initialize()
            list_result = await session.list_tools()
            mcp_tools = list_result.tools

    # Build Responses-flattened tools (default)
    function_tools, reverse_index = bridge_mcp_tools_to_function_tools(
        mcp_tools, server_label=SERVER_LABEL
    )

    print("\n--- OpenAI function tools for RESPONSES (flattened) ---")
    print(json.dumps(function_tools, indent=2, ensure_ascii=False))

    # Persist for reuse (required by other scripts)
    with open("context7_function_tools.responses.json", "w", encoding="utf-8") as f:
        json.dump(function_tools, f, indent=2, ensure_ascii=False)

    # Also dump the reverse index (used by your dispatcher)
    print("\n--- Reverse index (function name -> MCP tool) ---")
    print(json.dumps(reverse_index, indent=2, ensure_ascii=False))
    with open("context7_reverse_index.json", "w", encoding="utf-8") as f:
        json.dump(reverse_index, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())
