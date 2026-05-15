from pydantic import BaseModel, Field

from sentinel_audit.llm_outputs import AdversaryOutput, AuditReport, StrategistOutput, ValidatorOutput


class AuditState(BaseModel):
    # ===== Inputs =====
    workspace_path: str
    documentation_links: list[str] = Field(default_factory=list)
    raw_code: dict[str, str] = Field(default_factory=dict)
    extracted_docs: dict[str, str] = Field(default_factory=dict)
    file_hashes: dict[str, str] = Field(default_factory=dict)  # filename -> sha256 hash for integrity checks
    changed_files: dict[str, str] = Field(default_factory=dict)  # Initially empty, filled with old code

    # ===== Typed Agent Outputs =====
    strategist_output: StrategistOutput | None = None
    adversary_output: AdversaryOutput | None = None
    validator_output: ValidatorOutput | None = None
    final_report: AuditReport | None = None

    # ===== Runtime / Orchestration =====
    current_agent: str = "strategist"
    iteration_count: int = 0
    max_iterations: int = 10
    next_step: str = "run"
    human_approval_required: bool = False
    pending_action: str | None = None
    token_usage_total: int = 0
    audit_thread_id: str | None = None  # Deterministic identifier
