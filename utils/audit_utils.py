import logging
from pathlib import Path

from state import AuditState
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
# For now only Solidity files for LLM input, but can be easily extended to other relevant file types when needed
LLM_INPUT_FILE_EXTENSIONS = ["*.sol"]
PREFIXES_TO_IGNORE_COMMENTS = ("//", )  # can be extended to "#", "--", etc. for other languages


def normalize_code(code: str) -> str:
    """
    Cleans Solidity/Python code by removing trailing whitespace, empty lines,
    and lines that are purely single-line comments.
    """
    normalized = "\n".join(
        r for line in code.strip().splitlines()
        if (r := line.rstrip())  # Walrus Operator (Python 3.8+)
        and not r.lstrip().startswith(PREFIXES_TO_IGNORE_COMMENTS)
    )
    return normalized


def scan_repo_compute_file_hashes(state: AuditState) -> None:
    workspace = Path(state.workspace_path)

    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    for file_ext in LLM_INPUT_FILE_EXTENSIONS:
        for path in workspace.rglob(file_ext, case_sensitive=True):
            # Skip ignored directories
            if any(part in IGNORED_DIRS for part in path.parts):
                continue

            relative_path = str(path.relative_to(workspace))

            with open(path, "r", encoding="utf-8") as f:
                original_content = f.read()

            normalized_content = normalize_code(original_content)
            if not normalized_content:
                logger.warning(f"File {relative_path} is empty after normalization. Skipping.")
                continue

            # Store normalized code
            state.raw_code[relative_path] = normalized_content
            file_hash = sha256_hex(normalized_content.encode("utf-8"))
            state.file_hashes[relative_path] = file_hash
