import dataclasses
import json
import pathlib
import typing

import openai

import defs
import utils


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

SYSTEM_PROMPT = """You operate ONLY by calling the `bash` tool exactly once.
Rules:
- Return exactly ONE tool call to `bash` (no assistant prose in that message).
- Prefer Python HEREDOCs for multi-file edits.
- Within the `python` HEREDOC, you can simply use the command `python` to run code; no 
  need to use `python3`.
- Do not commit changes using `git`.
"""

SUMMARY_SYSTEM_PROMPT = (
    "You summarize tool execution results briefly and clearly for humans."
)


@dataclasses.dataclass
class CommandProposal:
    command: str
    timeout_sec: int


class BashToolCaller:
    """Stateful tool calling with the Responses API.

    Flow:
      1) responses.create(..., tools=[...], tool_choice=..., store=True)
      2) Extract function call (id + arguments), run the local tool
      3) responses.create(previous_response_id=..., input=[
             { "type": "function_call_output", "call_id": <func_call_id>, "output": "<JSON string>" },
             { "role": "system", "content": SUMMARY_SYSTEM_PROMPT },
             { "role": "user", "content": "Please summarize ..." }
         ], tools=[], tool_choice="none")
    """

    def __init__(self, model: str = "gpt-5"):
        self.client = openai.OpenAI()
        self.model = model
        self.task: typing.Optional[str] = None
        self._first_response_id: typing.Optional[str] = None
        self._last_tool_call_id: typing.Optional[str] = None

    def propose_command(self, task: str) -> CommandProposal:
        """Ask the model for exactly one bash function call and remember state."""
        self.task = task.strip()

        first = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task: {self.task}\n"
                        "Return exactly one bash tool call that accomplishes this."
                    ),
                },
            ],
            tools=[BASH_TOOL],
            tool_choice={"type": "function", "name": "bash"},  # force single call
            store=True,  # <-- keep server-side state for next turn
        )
        self._first_response_id = first.id

        # Note: the code block below is "unsafe" in that it assumes:
        #  * A function_call has been returned
        #  * `arguments` is a JSON string
        # we'll safen it up later
        func_item = next(item for item in first.output if item.type == "function_call")
        args = json.loads(func_item.arguments)

        # remember for function_call_output
        self._last_tool_call_id = func_item.call_id

        return CommandProposal(
            command=args["command"], timeout_sec=int(args.get("timeout_sec", 120))
        )

    def summarize_result(self, result: defs.CommandResult) -> str:
        """Submit the tool result via function_call_output and continue statefully."""
        if not self._first_response_id or not self._last_tool_call_id:
            raise RuntimeError("Missing response/call id; call propose_command() first.")

        # The Responses API expects the tool result as a `function_call_output` item
        # referencing the original function call's id via `call_id`. The `output`
        # should be a **string** (often a JSON string).
        tool_output_item = {
            "type": "function_call_output",
            "call_id": self._last_tool_call_id,  # the id from the function_call item
            "output": json.dumps(dataclasses.asdict(result)),
        }

        # Continue the *same* conversation state; include a concise summary request.
        # Disable tools here to ensure plain text.
        followup = self.client.responses.create(
            model=self.model,
            previous_response_id=self._first_response_id,
            input=[
                tool_output_item,
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Please summarize the tool result you just received. "
                        "Explain what was done and any user-facing changes, concisely."
                    ),
                },
            ],
            tools=[],  # no additional tools available in this step
            tool_choice="none",  # force text
        )

        return (getattr(followup, "output_text", "") or "").strip()


if __name__ == "__main__":
    repo = pathlib.Path("").resolve()
    task = """
        I'm interested in changing the name of this app to "GPT-Tree" from "GPTree". Any 
        chance you could try making that change in all necessary files? Please make sure
        you replace in all user-facing places in the repo that that name could appear.
        """

    agent = BashToolCaller(model="gpt-5")

    try:
        # 1) Get a single bash command, preserving state
        proposal = agent.propose_command(task)
        print("=== proposed bash command ===\n", proposal.command)

        # 2) Run it locally
        result = utils.run_bash(
            proposal.command, cwd=repo, timeout_sec=proposal.timeout_sec
        )
        print("\n=== stdout ===\n", result.stdout)
        if result.stderr:
            print("=== stderr ===\n", result.stderr)
        print("=== rc ===", result.returncode)

        # 3) Send function_call_output + ask for summary (stateful continuation)
        if result.returncode == 0:
            summary = agent.summarize_result(result)
            print("\n=== final explanation ===\n", summary)
        else:
            print(
                "\n=== final explanation ===\n",
                "The command failed to execute, so no summary was generated.",
            )

    except Exception as e:
        print(f"\nAn error occurred: {e}")
