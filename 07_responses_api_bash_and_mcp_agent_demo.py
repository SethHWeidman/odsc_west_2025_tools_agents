import asyncio
import dataclasses
import json
from os import environ
import pathlib
import typing

import mcp
from mcp.client import stdio
import openai

import utils


AGENT_SYSTEM_PROMPT = """You work in steps. At each step:
- Return at most ONE tool call (bash or an MCP tool).
- Prefer non-interactive bash commands and HEREDOCs for edits.
- After observing results, decide the next step.
- Stop proposing actions when the task is accomplished; then return plain text.
"""

SUMMARY_SYSTEM_PROMPT = "Summarize the overall results for the user."

# -------------------- Tools ---------------------------------------------------

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Execute a non-interactive bash command in the working directory.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_sec": {"type": "integer", "minimum": 1, "default": 120},
        },
        "required": ["command"],
    },
}

with open("context7_function_tools.responses.json", "r") as f:
    CONTEXT7_MCP_TOOLS = json.load(f)

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


class BashAndMcpAgent:
    """
    Multi-step agent loop using two tool families:
      - Local bash execution (function tool: `bash`)
      - Remote Context7 MCP tools (exposed as function tools via bridge JSON)

    Loop:
      • First call: responses.create(..., tools=[bash] + CONTEXT7_MCP_TOOLS, store=True)
      • If the model invokes a tool: execute it and return a function_call_output
      • If the model returns plain text (no tool calls): DONE
      • Final: optional natural-language summary with tools disabled
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_steps: int = 20,
        confirm: bool = False,
        verbose: bool = True,
        clip: int = 500,
        log_file: typing.Optional[str] = "agent_log.txt",
    ):
        self.client = openai.OpenAI()
        self.model = model
        self.max_steps = max_steps
        self.confirm = confirm
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
        # needed to add high reasoning effort to get the demo to work
        kwargs["reasoning"] = {"effort": 'high'}
        if self.response_id:
            kwargs["previous_response_id"] = self.response_id
        else:
            kwargs["store"] = True

        if expose_tools:
            kwargs["tools"] = [BASH_TOOL] + CONTEXT7_MCP_TOOLS
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tools"] = []
            kwargs["tool_choice"] = "none"

        resp = self.client.responses.create(**kwargs)
        self.response_id = resp.id
        return resp

    def _extract_function_calls(self, resp) -> list[dict]:
        """
        Collect all function_call items.
        Returns: [{'id': call_id, 'name': name, 'arguments': json_string}, ...]
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
        return [c for c in calls if c["id"] and c["name"] is not None]

    # --- public API ----------------------------------------------------------

    def run(self, task: str) -> str:
        """
        Minimal agent loop (bash + MCP):
        - Call model with bash and Context7 MCP tools available.
        - If a tool was invoked, execute and feed the observation back.
        - If no tool call is present, treat as DONE and return text.
        - Finally, ask for a brief summary (tools disabled).
        """
        self._log("▶️  Task:", task.strip())
        # First turn: ask the model; expose both tool families
        resp = self._send_to_model(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": task.strip()},
            ],
            expose_tools=True,
        )

        for step in range(1, self.max_steps + 1):
            self.step_count = step

            calls = self._extract_function_calls(resp)
            if calls:
                self._log(f"\n🔧 Step {step}: {len(calls)} function call(s) requested")

                outputs = []
                declined = False
                for c in calls:
                    self.tool_calls += 1
                    fn_name = c["name"]

                    # Parse function arguments
                    try:
                        args_obj = (
                            json.loads(c["arguments"])
                            if isinstance(c["arguments"], str)
                            else (c["arguments"] or {})
                        )
                    except Exception:
                        args_obj = {}

                    # Optional human-in-the-loop confirmation
                    if self.confirm:
                        if fn_name == "bash":
                            cmd_preview = args_obj.get("command", "")
                            print(f"\n[step {step}] propose bash:\n{cmd_preview}")
                        else:
                            print(
                                f"\n[step {step}] propose MCP: {fn_name} args={args_obj}"
                            )
                        if input("Execute? [Y/n] ").strip().lower() == "n":
                            # Feed brief guidance and let the model try again with tools
                            resp = self._send_to_model(
                                [
                                    {
                                        "role": "user",
                                        "content": "Declined. Try another approach.",
                                    }
                                ],
                                expose_tools=True,
                            )
                            declined = True
                            break

                    if fn_name == "bash":
                        command = args_obj.get("command", "")
                        timeout_sec = int(args_obj.get("timeout_sec", 120))
                        self._log(f"   bash → {command}")
                        result = utils.run_bash(
                            command, cwd=REPO, timeout_sec=timeout_sec
                        )
                        # Log outputs similar to the 04 script
                        self._log(f"   rc={result.returncode}")
                        if result.stdout:
                            self._log("   stdout:\n", self._clip(result.stdout))
                        if result.stderr:
                            self._log("   stderr:\n", self._clip(result.stderr))
                        tool_output_text = json.dumps(dataclasses.asdict(result))
                    else:
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
                            # Log a clipped view of MCP output for visibility
                            if tool_output_text:
                                self._log(
                                    "   mcp output (clipped):\n",
                                    self._clip(tool_output_text),
                                )

                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": c["id"],  # matches the model's call_id
                            "output": tool_output_text,  # must be a string
                        }
                    )

                if declined:
                    # Skip sending outputs; we already nudged the model. Continue loop.
                    continue

                # Send all outputs back into the SAME thread; keep tools exposed
                resp = self._send_to_model(outputs, expose_tools=True)
                continue

            # No tool call this turn => model emitted plain text → DONE
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
    agent = BashAndMcpAgent(
        verbose=True, confirm=False, log_file="agent_multi_tools_log.txt", max_steps=50
    )

    TASK = (
        """
        Please update the backend of this project to use FastAPI instead of Flask. Do 
        your work on an appropriately-named branch. Please ensure you review the 
        \"latest and greatest\" FastAPI docs. The goal is that, after your code changes, 
        I should be able to run `./start-local-gpt.sh` and it should "just work". 

        Few more important things: 

        1. Do not try too hard to test your changes; just tell me when you think you've 
           finished refactoring the back end and think it will work.
        2. Any files that are not under version control can be ignored; they do not 
           contribute to the app's functionality.
        3. Do not feel the need to `git add` or `git commit` files at the end.
        4. You are on a system that does not have an `applypatch` command. 
        5. Please ignore any files in the repo that have  not been  `git add`ed - do 
           not delete them.
        """
    ).strip()

    summary = agent.run(task=TASK)
    print("\n=== final summary ===\n", summary)
    print(f"\n(stats) steps={agent.step_count}, function_tool_calls={agent.tool_calls}")
