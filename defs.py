import dataclasses


@dataclasses.dataclass
class CommandResult:
    """Represents the result of executing a local bash command."""

    command: str
    returncode: int
    stdout: str
    stderr: str
