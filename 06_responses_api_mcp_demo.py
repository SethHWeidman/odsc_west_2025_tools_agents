import asyncio
import json
from os import environ
import pathlib
import typing

import mcp
from mcp.client import stdio
import openai


AGENT_SYSTEM_PROMPT = """You work in steps. At each step:
- Return at most ONE tool call.
- After observing results, decide the next step.
- Stop proposing actions when the task is accomplished; then return plain text.
"""

REFLECT_USER_PROMPT = (
    "Given the latest observation, propose the next single action (one MCP tool call) "
    "or return a final text answer if complete."
)

SUMMARY_SYSTEM_PROMPT = "Summarize the overall results for the user."

# -------------------- Tools ---------------------------------------------------

with open("context7_function_tools.responses.json", "r") as f:
    CONTEXT7_MCP_TOOLS = json.load(f)

# -------------------- reverse index (function -> MCP) -------------------------

with open("context7_reverse_index.json", "r") as f:
    REVERSE_INDEX = json.load(f)


REPO = pathlib.Path("").resolve()


# -------------------- Context7 MCP dispatcher ---------------------------------


class Context7Dispatcher:
    """
    Minimal dispatcher that starts the Context7 MCP server over stdio, calls a tool,
    and returns a text blob. For simplicity this spins up the MCP server per call.
    You can keep a persistent session if you prefer; this keeps the script simple.
    """

    def __init__(self, verbose: bool = True):
        self.api_key = environ.get("CONTEXT7_API_KEY", "")
        self.verbose = verbose

    def _vlog(self, *args):
        if self.verbose:
            print(*args)

    def _flatten_mcp_tool_result(self, result: typing.Any) -> str:
        """
        Flatten an MCP tool call result into a single printable string.

        Input
        -----
        result: Any
            The object returned by `session.call_tool(...)`. It typically has a
            `.content` attribute that is an iterable of content parts. Each part may
            expose one of the following attributes:
              - `text`: a plain string segment
              - `json` or `data`: a JSON-serializable Python object
              - otherwise: any object, which will be converted with `str(...)`

        Output
        ------
        str
            A single string formed by joining all recognized parts with newlines.
            JSON/data objects are serialized with `json.dumps(..., ensure_ascii=False,
            indent=2)` when possible; if serialization fails, `str(obj)` is used. If no
            recognizable content exists, returns an empty string.
        """
        parts: list[str] = []
        for c in getattr(result, "content", []) or []:
            # Handle common MCP content kinds: text, json, blob
            text = getattr(c, "text", None)
            if text is not None:
                parts.append(text)
                continue
            obj = getattr(c, "json", None) or getattr(c, "data", None)
            if obj is not None:
                try:
                    parts.append(json.dumps(obj, ensure_ascii=False, indent=2))
                except Exception:
                    parts.append(str(obj))
                continue
            # Fallback representation
            parts.append(str(c))
        return "\n".join(p for p in parts if p is not None)

    async def _acall(self, mcp_tool_name: str, arguments: dict | None) -> str:
        stdio_params = mcp.StdioServerParameters(
            command="npx",
            args=["-y", "@upstash/context7-mcp", "--api-key", self.api_key],
        )
        async with stdio.stdio_client(stdio_params) as (reader, writer):
            async with mcp.ClientSession(reader, writer) as session:
                await session.initialize()
                self._vlog(
                    f"   [dispatcher] calling MCP tool: {mcp_tool_name} "
                    f"args={arguments or {}}"
                )
                result = await session.call_tool(mcp_tool_name, arguments or {})
                out = self._flatten_mcp_tool_result(result)
                return out or ""

    def call(self, mcp_tool_name: str, arguments: dict | None) -> str:
        # Run in a new event loop. In async contexts, await `_acall` directly.
        return asyncio.run(self._acall(mcp_tool_name, arguments))


# -------------------- Agent ---------------------------------------------------


class DocsAgent:
    """
    Multi-step agent loop using only a remote MCP tool (Context7).

    Loop:
      • First call: responses.create(..., tools=[CONTEXT7_MCP], store=True)
      • If the model invokes an MCP tool: runtime executes it; we continue the loop
      • If the model returns plain text (no tool calls): DONE
      • Final: optional natural-language summary with tools disabled

    The MCP tool is executed by the Responses runtime; you do NOT write a local
    dispatcher for MCP. We keep passing previous_response_id so the thread stays
    stateful and the server's tool list stays cached.
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_steps: int = 20,
        verbose: bool = True,
        clip: int = 500,
        log_file: typing.Optional[str] = "agent_log.txt",
    ):
        self.client = openai.OpenAI()
        self.model = model
        self.max_steps = max_steps
        self.response_id: typing.Optional[str] = None
        self.verbose = verbose
        self.clip = clip
        self.step_count = 0
        self.tool_calls = 0

        # dispatcher for MCP function execution
        self.dispatcher = Context7Dispatcher(verbose=verbose)

        if log_file is None:
            self.log_file = None
        else:
            log_path = pathlib.Path(log_file)
            if not log_path.is_absolute():
                log_path = REPO / log_path
            self.log_file = log_path

    # --- logging helpers -----------------------------------------------------

    def _log(self, *args, **kwargs):
        if not self.verbose:
            return
        print(*args, **kwargs)
        if not self.log_file:
            return
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        text = sep.join(str(arg) for arg in args) + end
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if kwargs.get("flush"):
                f.flush()

    def _clip(self, s: typing.Optional[str]) -> str:
        s = "" if s is None else (s if isinstance(s, str) else str(s))
        return s if len(s) <= self.clip else s[: self.clip] + "…"

    # --- API helper ----------------------------------------------------------

    def _send_to_model(self, input_items, expose_tools: bool = True):
        """Call Responses API, continuing the same server-side conversation."""
        kwargs = dict(model=self.model, input=input_items)
        if self.response_id:
            kwargs["previous_response_id"] = self.response_id
        else:
            kwargs["store"] = True

        if expose_tools:
            kwargs["tools"] = CONTEXT7_MCP_TOOLS
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tools"] = []
            kwargs["tool_choice"] = "none"

        resp = self.client.responses.create(**kwargs)
        self.response_id = resp.id
        return resp

    def _extract_function_calls(self, resp) -> list[dict]:
        """
        Collect **function_call** items (not 'tool_call').
        Returns [{'id': call_id, 'name': name, 'arguments': json_string}, ...]
        """
        calls = []
        for it in getattr(resp, "output", []) or []:
            if getattr(it, "type", "") == "function_call":
                calls.append(
                    {
                        "id": getattr(it, "call_id", None) or getattr(it, "id", None),
                        "name": getattr(it, "name", None),
                        "arguments": getattr(it, "arguments", "") or "",
                    }
                )
        # Filter out any malformed entries
        return [c for c in calls if c["id"] and c["name"] is not None]

    # --- public API ----------------------------------------------------------

    def run(self, task: str) -> str:
        """
        Minimal agent loop (MCP-only):
        - Call model with Context7 MCP available.
        - If an MCP tool was invoked, continue the loop and ask for the next step.
        - If no tool call is present, treat as DONE and return text.
        - Finally, ask for a brief summary (tools disabled).
        """
        self._log("▶️  Task:", task.strip())
        # First turn: ask the model; expose bridged function tools
        resp = self._send_to_model(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": task.strip()},
            ],
            expose_tools=True,
        )

        for step in range(1, self.max_steps + 1):
            self.step_count = step

            # 1) Did the model ask to call any tools?
            calls = self._extract_function_calls(resp)
            if calls:
                self._log(f"\n🔧 Step {step}: {len(calls)} function call(s) requested")

                # Build ONLY function_call_output items (one per call) as the next input
                outputs = []
                for c in calls:
                    self.tool_calls += 1
                    fn_name = c["name"]

                    # Parse function arguments (often JSON string)
                    try:
                        args_obj = (
                            json.loads(c["arguments"])
                            if isinstance(c["arguments"], str)
                            else (c["arguments"] or {})
                        )
                    except Exception:
                        args_obj = {}

                    mapping = REVERSE_INDEX.get(fn_name)
                    if not mapping:
                        tool_output_text = (
                            "Unknown tool: "
                            + fn_name
                            + ". Available: "
                            + ", ".join(REVERSE_INDEX.keys())
                        )
                    else:
                        mcp_tool = mapping["mcp_name"]
                        tool_output_text = self.dispatcher.call(mcp_tool, args_obj)

                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": c["id"],  # MUST match the model's call_id
                            "output": tool_output_text,  # MUST be a string
                        }
                    )

                # Send all outputs back into the SAME thread; keep tools exposed
                resp = self._send_to_model(outputs, expose_tools=True)
                continue

            # No MCP tool call this turn => model emitted plain text → DONE
            last_text = getattr(resp, "output_text", "") or ""
            self._log(
                f"✅ Done signal at step {step}: model returned no tool call.\n"
                f"   Last assistant text (clipped):\n{self._clip(last_text)}"
            )
            break
        else:
            return "Stopped: step limit reached."

        # Final: concise summary (tools disabled)
        self._log(
            f"\nℹ️  Totals → steps: {self.step_count}, function calls executed: "
            f"{self.tool_calls}"
        )
        summary = self._send_to_model(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Summarize the overall results as bullet points for the "
                    "user.",
                },
            ],
            expose_tools=False,
        )
        return (getattr(summary, "output_text", "") or "").strip()


# -------------------- Demo ----------------------------------------------------

if __name__ == "__main__":
    agent = DocsAgent(verbose=True, log_file="agent_mcp_log.txt")

    TASK = (
        "Tell me about the npm package documentation for Express.js. Please make sure "
        "you tell me the \"latest and greatest\" info."
    ).strip()

    summary = agent.run(task=TASK)
    print("\n=== final summary ===\n", summary)
    print(f"\n(stats) steps={agent.step_count}, function_tool_calls={agent.tool_calls}")
