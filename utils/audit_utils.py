import logging
from pathlib import Path

from agents.audit_state import AuditState
from utils.cryptography_utils import sha256_hex

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    "node_modules",
    "lib",
    "artifacts",
    "cache",
    ".git",
    "out",
    "build",
}


def normalize_solidity_code(code: str) -> str:
    """
    Cleans Solidity code by removing trailing whitespace, empty lines,
    and lines that are purely single-line comments.
    """
    normalized = "\n".join(
        r for line in code.strip().splitlines()
        if (r := line.rstrip())  # Walrus Operator (Python 3.8+)
        and not r.lstrip().startswith("//")  # Skip single-line comments
    )
    return normalized


def scan_repo_compute_file_hashes(state: AuditState) -> None:
    workspace = Path(state.workspace_path)

    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    for path in workspace.rglob("*.sol"):
        # Skip ignored directories
        if any(part in IGNORED_DIRS for part in path.parts):
            continue

        relative_path = str(path.relative_to(workspace))

        with open(path, "r", encoding="utf-8") as f:
            original_content = f.read()

        normalized_content = normalize_solidity_code(original_content)
        if not normalized_content:
            logger.warning(f"File {relative_path} is empty after normalization. Skipping.")
            continue

        # Store normalized code
        state.raw_code[relative_path] = normalized_content
        file_hash = sha256_hex(normalized_content.encode("utf-8"))
        state.file_hashes[relative_path] = file_hash
