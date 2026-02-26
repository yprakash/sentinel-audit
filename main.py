import argparse
import asyncio
import logging
import shutil
from pathlib import Path

from git import Repo

from agents.audit_state import AuditState
from utils.audit_utils import scan_repo_compute_file_hashes
from utils.cryptography_utils import sha256_hex

logger = logging.getLogger('main')


def compute_audit_thread_id(user: str, state: AuditState) -> str:
    """
    Compute deterministic audit thread ID from sorted file hashes.
    """
    if not user or not state.file_hashes:
        raise ValueError("User and file hashes must be provided to compute audit thread ID.")

    sorted_items = sorted(state.file_hashes.items())  # Ensure deterministic ordering

    combined = f"user:{user}"
    for filename, file_hash in sorted_items:
        combined += f"\n{filename}:{file_hash}"

    return sha256_hex(combined.encode("utf-8"))


# =========================
# Bootstrap Logic
# =========================
def audit_bootstrap(target: str, docs_url: str = None) -> AuditState:
    work_dir: Path = Path("./workdir")
    if work_dir.exists():
        logger.info(f"Deleting existing local directory: {work_dir}")
        shutil.rmtree(work_dir)  # Clean start
    work_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created local directory: {work_dir}")

    # Handle GitHub vs Local
    if target.startswith("https://github.com"):
        logger.info(f"Cloning remote repository: {target}")
        Repo.clone_from(target, work_dir)
    else:
        logger.info(f"Using local directory: {target}")
        shutil.copytree(target, work_dir, dirs_exist_ok=True)

    state: AuditState = AuditState(
        workspace_path=work_dir.stem,
        documentation_links=[docs_url] if docs_url else []
    )
    scan_repo_compute_file_hashes(state)
    return state


async def run_audit(user: str, target: str, docs_url: str = None):
    # 1. Initialize project (Clone GitHub or verify local path)
    logger.info(f"Initializing audit for: {target}")

    # 2. Define Initial State
    initial_state: AuditState = audit_bootstrap(target, docs_url=docs_url)
    audit_uid = compute_audit_thread_id(user, initial_state)

    # 3. Invoke LangGraph (Stateful Execution)
    # Using a config with thread_id for persistence/HITL
    config = {"configurable": {"thread_id": audit_uid}}


async def main():
    parser = argparse.ArgumentParser(description="Sentinel-Audit CLI")
    parser.add_argument("--user", required=True, help="username or unique_id for audit tracking")
    parser.add_argument("--target", required=True, help="GitHub URL or local directory")
    parser.add_argument("--docs", help="URL to documentation/whitepaper")

    args = parser.parse_args()

    shutdown_event = asyncio.Event()
    try:
        await run_audit(args.user, args.target, args.docs)
    # except KeyboardInterrupt:
    finally:
        shutdown_event.set()


if __name__ == "__main__":
    asyncio.run(main())
