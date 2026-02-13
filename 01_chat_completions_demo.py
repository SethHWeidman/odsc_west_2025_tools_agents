import dataclasses
import json
import pathlib

import openai

import defs
import utils

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a non-interactive bash command in the working "
        "directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_sec": {"type": "integer", "minimum": 1, "default": 120},
            },
            "required": ["command"],
        },
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


class BashToolCaller:
    """A tool caller that uses a bash tool to accomplish tasks."""

    def __init__(self, model: str = "gpt-5"):
        self.client = openai.OpenAI()
        self.model = model
        self.messages = []

    def propose_command(self, task: str) -> defs.CommandProposal:
        """Given a task, proposes a single bash command to accomplish it, using the OpenAI."""
        self._create_initial_messages(task)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=[BASH_TOOL],
            tool_choice={"type": "function", "function": {"name": "bash"}},
        )
        assistant_msg = resp.choices[0].message
        self.messages.append(assistant_msg)

        if not assistant_msg.tool_calls:
            raise ValueError("The model did not return a tool call.")

        tool_call = assistant_msg.tool_calls[0]
        args = json.loads(tool_call.function.arguments or "{}")

        return defs.CommandProposal(
            command=args["command"],
            timeout_sec=int(args.get("timeout_sec", 120)),
            tool_call_id=tool_call.id,
        )

    def summarize_result(
        self, proposal: defs.CommandProposal, result: defs.CommandResult
    ) -> str:
        """Summarizes the outcome of the executed command."""
        self._add_tool_result_to_history(proposal, result)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=[BASH_TOOL],
            tool_choice="none",
        )
        return resp.choices[0].message.content

    def _create_initial_messages(self, task: str):
        """Helper to set up the initial message list for a new task."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Task: {task}\nReturn exactly one bash tool call that "
                "accomplishes this.",
            },
        ]

    def _add_tool_result_to_history(
        self, proposal: defs.CommandProposal, result: defs.CommandResult
    ):
        """Helper to append the tool execution results to the message history."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": proposal.tool_call_id,
                "name": "bash",
                "content": json.dumps(dataclasses.asdict(result)),
            }
        )
        self.messages.append(
            {
                "role": "user",
                "content": "Based on the tool result, provide a brief summary of what "
                "was done, including any changes made.",
            }
        )


if __name__ == "__main__":
    repo = pathlib.Path("").resolve()
    task = """
        I'm interested in changing the name of this app to "GPT-Tree" from "GPTree". Any 
        chance you could try making that change in all necessary files? Please make sure
        to only change it in places where a user would see the text of the old name 
        appear.
        """

    agent = BashToolCaller(model="gpt-5.2")

    try:
        # 1. Propose a command
        proposal = agent.propose_command(task)
        print("=== proposed bash command ===\n", proposal.command)

        # 2. Execute the command
        result = utils.run_bash(
            proposal.command, cwd=repo, timeout_sec=proposal.timeout_sec
        )
        print("\n=== stdout ===\n", result.stdout)
        if result.stderr:
            print("=== stderr ===\n", result.stderr)
        print("=== rc ===", result.returncode)

        # 3. Summarize the result
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
