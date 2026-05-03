import argparse
import asyncio
import logging
import shutil
import time
from pathlib import Path

from dotenv import load_dotenv
from git import Repo

from utils.initializer import init
from utils.shutdown_manager import shutdown_event, shutdown_manager

load_dotenv()  # populate keys before importing classes (instantiated)

from sentinel_audit.graph import get_graph
from sentinel_audit.state import AuditState
from sentinel_audit.audit_utils import scan_repo_compute_file_hashes
from utils.cryptography_utils import sha256_hex

logger = logging.getLogger('main')


def compute_audit_thread_id(userid: str, state: AuditState) -> str:
    """
    Compute deterministic audit thread ID from sorted file hashes.
    """
    start = time.perf_counter()
    if not userid or not state.file_hashes:
        raise ValueError("User and file hashes must be provided to compute audit thread ID.")

    sorted_items = sorted(state.file_hashes.items())  # Ensure deterministic ordering

    combined = f"user:{userid}"
    for filename, file_hash in sorted_items:
        combined += f"\n{filename}:{file_hash}"

    result = sha256_hex(combined.encode("utf-8"))
    logger.info("Computed audit thread_id in %.3f seconds: %s", time.perf_counter() - start, result)
    return result


def audit_bootstrap(target: str, docs_url: str = None) -> AuditState:
    work_dir: Path = Path("./workdir")
    if work_dir.exists():
        logger.warning("Deleting existing local directory: %s", work_dir)
        shutil.rmtree(work_dir)  # Clean start
    work_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Created local directory: %s", work_dir)

    # Handle GitHub vs Local
    if target.startswith("https://github.com"):
        logger.info("Cloning remote repository: %s", target)
        Repo.clone_from(target, work_dir)
    else:
        if not Path(target).exists():
            raise FileNotFoundError(f"File {target} does not exist.")
        logger.info("Using local directory: %s", target)
        shutil.copytree(target, work_dir, dirs_exist_ok=True)

    state: AuditState = AuditState(
        workspace_path=work_dir.stem,
        documentation_links=[docs_url] if docs_url else []
    )
    scan_repo_compute_file_hashes(state)
    return state


async def run_audit(userid: str, target: str, thread_id: str = None, docs_url: str = None):
    graph = await get_graph()
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Resuming audit for thread %s", thread_id)
    else:
        # 1. Initialize project (Clone GitHub or verify local path) & Define Initial State
        logger.info("Initializing audit for: %s", target)
        initial_state: AuditState = audit_bootstrap(target, docs_url=docs_url)
        thread_id = compute_audit_thread_id(userid, initial_state)
        logger.info("Audit thread ID: %s generated for user: %s with %d files.",
                    thread_id, userid, len(initial_state.file_hashes))

        initial_state.audit_thread_id = thread_id

        # 2. Invoke LangGraph (Stateful Execution)
        # Using a config with thread_id for persistence/HITL
        config = {"configurable": {"thread_id": thread_id}}
        logger.info("Starting audit for thread %s", thread_id)

        await graph.ainvoke(initial_state, config=config)

    final_state = None
    try:
        while True:
            state = await graph.aget_state(config)
            if not state.next:
                break
            print(f"Paused before: {state.next}")

            user_input = input("Continue? (y/n): ").strip().lower()
            if user_input != "y":
                print("Execution aborted by user")
                break
            # Resume execution
            await graph.ainvoke(None, config=config)

        final_state = await graph.aget_state(config)
        logger.info("Agentic audit run completed for user %s, thread %s", userid, thread_id)
        return final_state
    except Exception:
        logger.exception("Failed to run audit for thread ID: %s", thread_id)
    finally:
        shutdown_event.set()
        await shutdown_manager.shutdown()

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

    init(Path(__file__).parent.name)
    try:
        await run_audit(args.user_id, args.target, args.thread_id, args.docs)
    except KeyboardInterrupt:
        logger.info("Ctrl+C Shutting down...")
    finally:
        shutdown_event.set()
        await shutdown_manager.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
