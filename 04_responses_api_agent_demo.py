import dataclasses
import json
import pathlib
import typing
import openai

import utils


AGENT_SYSTEM_PROMPT = """You work in steps. At each step:
- Return at most ONE tool call to `bash`.
- Prefer non-interactive commands and Python HEREDOCs for edits.
- After observing results, decide the next step. Avoid git commits.
- Stop proposing actions when the task is accomplished.
"""

REFLECT_USER_PROMPT = (
    "Given the latest observation, propose the next single bash command."
)

SUMMARY_SYSTEM_PROMPT = "Summarize the overall changes and user-facing results."


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

REPO = pathlib.Path("").resolve()


@dataclasses.dataclass
class Proposal:
    command: str
    timeout_sec: int
    call_id: str


class BashAgent:
    """
    Multi-step agent loop using the Responses API.
    - First call: create(..., tools=[bash], store=True)
    - Each step: execute returned function_call, send function_call_output,
                 then continue with previous_response_id=..., still exposing the tool.
    - Finish: when done_check(...) returns True, request a final summary (tools
      disabled).
    """

    def __init__(
        self,
        model: str = "gpt-5",
        max_steps: int = 20,
        confirm: bool = False,
        verbose: bool = True,
        clip: int = 400,
    ):
        self.client = openai.OpenAI()
        self.model = model
        self.max_steps = max_steps
        self.confirm = confirm
        self.response_id: typing.Optional[str] = None
        self.last_call_id: typing.Optional[str] = None
        self.verbose = verbose
        self.clip = clip
        self.step_count = 0
        self.tool_calls = 0

    # --- model IO helpers ----------------------------------------------------

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _clip(self, s: typing.Optional[str]) -> str:
        s = "" if s is None else (s if isinstance(s, str) else str(s))
        return s if len(s) <= self.clip else s[: self.clip] + "…"

    def _first_or_next(self, input_items, expose_tools: bool = True) -> typing.Any:
        """Call Responses API, continuing the same server-side conversation."""
        kwargs = dict(model=self.model, input=input_items)
        if self.response_id:
            kwargs["previous_response_id"] = self.response_id
        else:
            kwargs["store"] = True

        if expose_tools:
            kwargs["tools"] = [BASH_TOOL]
        else:
            kwargs["tools"] = []
            kwargs["tool_choice"] = "none"

        resp = self.client.responses.create(**kwargs)
        self.response_id = resp.id
        return resp

    @staticmethod
    def _extract_single_function_call(resp) -> Proposal | None:
        """Return the single bash call if present; otherwise None (model emitted
        text)."""

        fc = next(
            (i for i in resp.output if getattr(i, "type", "") == "function_call"), None
        )
        if not fc:
            return None

        args = json.loads(getattr(fc, "arguments", "{}"))
        call_id = getattr(fc, "call_id", None)
        if not call_id:
            raise ValueError("Missing call_id on function_call item")

        return Proposal(
            command=args["command"],
            timeout_sec=int(args.get("timeout_sec", 120)),
            call_id=call_id,
        )

    # --- public API ----------------------------------------------------------

    def run(self, task: str) -> str:
        """
        Minimal agent loop:
        - Keep calling the model with the bash tool available.
        - Execute exactly one function call if present.
        - If the model returns no function_call, we're DONE.
        - Then request a final natural-language summary (no tools).
        """

        # Seed the conversation (tools enabled)
        self._log("▶️  Task:", task.strip())
        resp = self._first_or_next(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": task.strip()},
            ]
        )
        for step in range(1, self.max_steps + 1):
            self.step_count = step
            proposal = self._extract_single_function_call(resp)

            # DONE: the model did not request a tool call this turn
            if proposal is None:
                last_text = getattr(resp, "output_text", "") or ""
                self._log(
                    f"✅ Done signal at step {step}: model returned no tool call.\n"
                    f"   Last assistant text (clipped):\n{self._clip(last_text)}"
                )
                break

            # Optional human-in-the-loop
            if self.confirm:
                print(f"\n[step {step}] propose:\n{proposal.command}")
                if input("Execute? [Y/n] ").strip().lower() == "n":
                    # Feed brief guidance and let the model try again with tools
                    resp = self._first_or_next(
                        [{"role": "user", "content": "Declined. Try another approach."}]
                    )
                    continue

            # Execute locally
            self.tool_calls += 1
            self._log(f"\n🔧 Step {step}: bash → {proposal.command}")
            result = utils.run_bash(
                proposal.command, cwd=REPO, timeout_sec=proposal.timeout_sec
            )

            self._log(f"   rc={result.returncode}")
            if result.stdout:
                self._log("   stdout:\n", self._clip(result.stdout))
            if result.stderr:
                self._log("   stderr:\n", self._clip(result.stderr))

            # Send observation back into the SAME Responses thread; keep tools enabled
            tool_output_item = {
                "type": "function_call_output",
                "call_id": proposal.call_id,  # must match function_call.call_id
                "output": json.dumps(dataclasses.asdict(result)),
            }
            resp = self._first_or_next([tool_output_item])

        else:
            # Hit step limit without ever seeing a non-tool reply
            return "Stopped: step limit reached."

        # Final: ask for a concise summary (tools disabled)
        self._log(
            f"\nℹ️  Totals → steps: {self.step_count}, tool calls executed: "
            f"{self.tool_calls}"
        )
        summary = self._first_or_next(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Summarize the overall changes and any user-facing "
                    "effects.",
                },
            ],
            expose_tools=False,
        )
        return (getattr(summary, "output_text", "") or "").strip()


if __name__ == "__main__":
    bash_agent = BashAgent(verbose=True)

    task = """
        I'm interested in changing the name of this app to "GPT-Tree" from "GPTree". Any 
        chance you could try making that change in all necessary files? Please make sure
        you replace in all user-facing places in the repo that that name could appear.
        """

    summary = bash_agent.run(task=task)
    print("\n=== final summary ===\n", summary)
    print(f"\n(stats) steps={bash_agent.step_count}, tool_calls={bash_agent.tool_calls}")
