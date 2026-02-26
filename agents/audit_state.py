from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class Finding(BaseModel):
    title: str
    severity: str  # High, Medium, Low, Info
    description: str
    code_snippet: str
    recommendation: str
    is_verified: bool = False


class AuditState(BaseModel):
    # ===== Inputs =====
    workspace_path: str
    documentation_links: List[str] = Field(default_factory=list)

    # ===== Shared Knowledge =====
    raw_code: Dict[str, str] = Field(default_factory=dict)
    file_hashes: Dict[str, str] = Field(default_factory=dict)  # filename -> sha256 hash for integrity checks
    changed_files: Dict[str, str] = Field(default_factory=dict)  # Initially empty, filled with old code
    extracted_docs: Dict[str, str] = Field(default_factory=dict)

    invariants: Dict[str, List[str]] = Field(default_factory=lambda: {
        "explicit": [],
        "implicit": [],
        "economic": [],
        "access_control": [],
        "state_transition": []
    })

    findings: List[Finding] = Field(default_factory=list)

    # ===== Execution Control =====
    current_agent: str = "strategist"
    iteration_count: int = 0
    max_iterations: int = 10

    human_approval_required: bool = False
    pending_action: Optional[str] = None

    token_usage_total: int = 0
    next_step: str = "run"
