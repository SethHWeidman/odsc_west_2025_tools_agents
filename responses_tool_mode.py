import dataclasses
import json
import pathlib
import subprocess
import typing

import openai

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


@dataclasses.dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


class BashToolCaller:
    """A tool caller that uses a bash tool to accomplish tasks, using the OpenAI
    Responses API)."""

    def __init__(self, model: str = "gpt-5"):
        self.client = openai.OpenAI()
        self.model = model
        self.task: typing.Optional[str] = None

    def propose_command(self, task: str) -> CommandProposal:
        """Given a task, proposes a single bash command to accomplish it."""
        self.task = task.strip()

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Task: {self.task}\n Return exactly one bash tool call that "
                        "accomplishes this."
                    ),
                },
            ],
            tools=[BASH_TOOL],
            # forces a single call
            tool_choice={"type": "function", "name": "bash"},
        )

        # Note: the code block below is "unsafe" in that it assumes:
        #  * A function_call has been returned
        #  * `arguments` is a JSON string
        # we'll safen it up later
        func_item = next(
            item for item in response.output if item.type == "function_call"
        )
        args = json.loads(func_item.arguments)

        return CommandProposal(
            command=args["command"], timeout_sec=int(args.get("timeout_sec", 120))
        )

    def run_bash(
        self, command: str, cwd: pathlib.Path, timeout_sec: int = 120
    ) -> CommandResult:
        """Executes a bash command and captures its output."""
        cp = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return CommandResult(
            command=command, returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr
        )

    def summarize_result(self, proposal: CommandProposal, result: CommandResult) -> str:
        """Summarizes the outcome of the executed command (no tools used here)."""
        result_json = json.dumps(dataclasses.asdict(result), indent=2)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Based on the tool result below, provide a brief summary of "
                        "what was done, including any changes made.\n\n"
                        f"Original task: {self.task or ''}\n"
                        f"Proposed command: {proposal.command}\n"
                        f"Tool execution result (JSON):\n {result_json}"
                    ),
                },
            ],
        )

        return getattr(response, "output_text", "").strip()


if __name__ == "__main__":
    repo = pathlib.Path("").resolve()
    task = """
        I'm interested in changing the name of this app to "GPT-Tree" from "GPTree". Any 
        chance you could try making that change in all necessary files? Please make sure
        you replace in all user-facing places in the repo that that name could appear.
        """
    agent = BashToolCaller(model="gpt-5")

    try:
        proposal = agent.propose_command(task)
        print("=== proposed bash command ===\n", proposal.command)

        result = agent.run_bash(
            proposal.command, cwd=repo, timeout_sec=proposal.timeout_sec
        )
        print("\n=== stdout ===\n", result.stdout)
        if result.stderr:
            print("=== stderr ===\n", result.stderr)
        print("=== rc ===", result.returncode)

        if result.returncode == 0:
            summary = agent.summarize_result(proposal, result)
            print("\n=== final explanation ===\n", summary)
        else:
            print(
                "\n=== final explanation ===\n",
                "The command failed to execute, so no summary was generated.",
            )

    except Exception as e:
        print(f"\nAn error occurred: {e}")
