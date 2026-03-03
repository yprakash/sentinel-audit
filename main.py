import argparse
import asyncio
import logging
import shutil
from pathlib import Path

from git import Repo

from graph import get_graph
from state import AuditState
from utils.audit_utils import scan_repo_compute_file_hashes
from utils.cryptography_utils import sha256_hex

logger = logging.getLogger('main')


def compute_audit_thread_id(userid: str, state: AuditState) -> str:
    """
    Compute deterministic audit thread ID from sorted file hashes.
    """
    if not userid or not state.file_hashes:
        raise ValueError("User and file hashes must be provided to compute audit thread ID.")

    sorted_items = sorted(state.file_hashes.items())  # Ensure deterministic ordering

    combined = f"user:{userid}"
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
        if not Path(target).exists():
            raise FileNotFoundError(f"File {target} does not exist.")
        logger.info(f"Using local directory: {target}")
        shutil.copytree(target, work_dir, dirs_exist_ok=True)

    state: AuditState = AuditState(
        workspace_path=work_dir.stem,
        documentation_links=[docs_url] if docs_url else []
    )
    scan_repo_compute_file_hashes(state)
    return state


async def run_audit(userid: str, target: str, thread_id: str = None, docs_url: str = None):
    # 1. Initialize project (Clone GitHub or verify local path)
    logger.info(f"Initializing audit for: {target}")

    # 2. Define Initial State
    initial_state: AuditState = audit_bootstrap(target, docs_url=docs_url)
    if thread_id:
        logger.info(f"Resuming existing audit session with thread ID: {thread_id}")
    else:
        thread_id = compute_audit_thread_id(userid, initial_state)
        logger.info(
            f"Audit thread ID: {thread_id} generated for user: {userid} with {len(initial_state.file_hashes)} files.")

    initial_state.audit_thread_id = thread_id

    # 3. Invoke LangGraph (Stateful Execution)
    # Using a config with thread_id for persistence/HITL
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()
    logger.info(f"Starting audit thread {thread_id}")

    final_state = await graph.ainvoke(initial_state, config=config)

    logger.info(f"Audit {thread_id} completed")

    return final_state


async def main():
    parser = argparse.ArgumentParser(description="Sentinel-Audit CLI")
    parser.add_argument("--user-id", help="username or unique_id for audit tracking")
    parser.add_argument("--thread-id", help="Resume a specific audit session")
    parser.add_argument("--target", required=True, help="GitHub URL or local directory")
    parser.add_argument("--docs", help="URL to documentation/whitepaper")

    args = parser.parse_args()

    if not args.user_id and not args.thread_id:  # Validation logic
        parser.error("At least one of --user-id or --thread-id must be provided.")

    shutdown_event = asyncio.Event()
    try:
        await run_audit(args.user_id, args.target, args.thread_id, args.docs)
    # except KeyboardInterrupt:
    finally:
        shutdown_event.set()


if __name__ == "__main__":
    asyncio.run(main())
