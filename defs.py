import dataclasses
import typing


@dataclasses.dataclass
class CommandProposal:
    """Represents a command proposed by the AI."""

    command: str
    timeout_sec: int
    tool_call_id: typing.Optional[str] = None


@dataclasses.dataclass
class CommandResult:
    """Represents the result of executing a local bash command."""

    command: str
    returncode: int
    stdout: str
    stderr: str
