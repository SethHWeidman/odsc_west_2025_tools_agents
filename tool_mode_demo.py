import json, pathlib, subprocess

import openai

client = openai.OpenAI()

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
- Target bash 3.2 compatibility: do NOT use `mapfile`. Prefer Python HEREDOCs for 
  multi-file edits.
- Within the `python` HEREDOC, you can simply use the command `python` to run code; no 
  need to use `python3`.
"""


def _user_prompt(task: str) -> str:
    return f"Task: {task}\nReturn exactly one bash tool call that accomplishes this."


def propose_bash_command(task: str, model="gpt-5"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(task)},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[BASH_TOOL],
        tool_choice={"type": "function", "function": {"name": "bash"}},
    )
    msg = resp.choices[0].message
    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments or "{}")
    return messages, msg, call.id, args["command"], int(args.get("timeout_sec", 120))


def run_bash(command: str, cwd: pathlib.Path, timeout_sec: int = 120):
    cp = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return {
        "command": command,
        "returncode": cp.returncode,
        "stdout": cp.stdout,
        "stderr": cp.stderr,
    }


def finalize_with_observation(
    messages, assistant_msg, tool_call_id, tool_result, model="gpt-5"
):
    messages = list(messages)
    messages.append(
        {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": assistant_msg.tool_calls,
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "bash",
            "content": json.dumps(tool_result),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": "Based on the tool result, provide a brief summary of what was "
            "done, including any changes made.",
        }
    )
    resp = client.chat.completions.create(
        model=model, messages=messages, tools=[BASH_TOOL], tool_choice="none"
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    repo = pathlib.Path("").resolve()
    task = """
        I'm interested in changing the name of this app to "GPT-Tree" from "GPTree". Any 
        chance you could try making that change in all necessary files? Please make sure
        you replace in all user-facing places in the repo that that name could appear.
        """

    messages, assistant_msg, call_id, command, tmo = propose_bash_command(task)
    print("\n=== proposed bash command ===\n", command)
    result = run_bash(command, cwd=repo, timeout_sec=tmo)

    print("=== stdout ===\n", result["stdout"])
    print("=== stderr ===\n", result["stderr"])
    print("=== rc ===", result["returncode"])

    final = finalize_with_observation(messages, assistant_msg, call_id, result)
    print("\n=== final explanation ===\n", final)
