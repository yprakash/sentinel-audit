from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# Core Enums used by sentinel-audit
class Severity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Confidence(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Category(str, Enum):
    state = "state"
    economic = "economic"
    access_control = "access_control"
    arithmetic = "arithmetic"
    reentrancy = "reentrancy"
    oracle = "oracle"
    liquidation = "liquidation"
    governance = "governance"
    dos = "dos"
    upgradeability = "upgradeability"
    initialization = "initialization"
    signature = "signature"
    randomness = "randomness"
    other = "other"


class ValidationStatus(str, Enum):
    confirmed = "confirmed"
    false_positive = "false_positive"
    inconclusive = "inconclusive"


class Exploitability(str, Enum):
    theoretical = "theoretical"
    practical = "practical"
    trivial = "trivial"


# Shared Core Models
class SourceLocation(BaseModel):
    file: str
    contract: Optional[str] = None
    function: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class Evidence(BaseModel):
    description: str
    code_snippet: Optional[str] = None
    stack_trace: Optional[str] = None
    location: Optional[SourceLocation] = None


class Invariant(BaseModel):
    id: str
    title: str
    description: str
    category: Category
    severity: Severity
    confidence: Confidence
    affected_contracts: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    trust_boundaries: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)


# Strategist Agent Output
class StrategistOutput(BaseModel):
    business_logic_summary: str
    protocol_overview: Optional[str] = None
    invariants: List[Invariant]
    assumptions: List[str]
    trust_boundaries: List[str]
    critical_components: List[str] = Field(default_factory=list)
    external_dependencies: List[str] = Field(default_factory=list)


# Adversary Agent Models
## Threat Model
class ThreatActor(BaseModel):
    name: str
    capabilities: List[str]
    assumptions: List[str]


class ThreatModel(BaseModel):
    attack_surface: List[str]
    threat_actors: List[ThreatActor]
    trust_assumptions: List[str]


## Forge Test Case
class ForgeTestCase(BaseModel):
    test_id: str
    invariant_id: str
    title: str
    description: str
    forge_test_code: str
    exploit_scenario: str
    expected_failure_mode: str
    attack_steps: List[str]
    required_setup: List[str] = Field(default_factory=list)


## Adversary Output
class AdversaryOutput(BaseModel):
    threat_model: ThreatModel
    test_cases: List[ForgeTestCase]
    attack_chains: List[str] = Field(default_factory=list)
    exploit_hypotheses: List[str] = Field(default_factory=list)


# Validator Agent Models
## Test Result
class TestExecutionResult(BaseModel):
    test_id: str
    passed: bool
    gas_used: Optional[int] = None
    execution_time_ms: Optional[int] = None
    logs: List[str] = Field(default_factory=list)
    stack_trace: Optional[str] = None


## Validated Finding
class ValidatedFinding(BaseModel):
    finding_id: str
    invariant_id: Optional[str] = None
    title: str
    description: str
    severity: Severity
    confidence: Confidence
    category: Category
    validation_status: ValidationStatus
    exploitability: Exploitability
    affected_contracts: List[str]
    impacted_functions: List[str]
    attack_scenario: str
    root_cause: str
    impact: str
    proof_of_concept: Optional[str] = None
    reproducible: bool = True
    reproducibility_notes: Optional[str] = None
    evidence: List[Evidence] = Field(default_factory=list)


## Validator Output
class ValidatorOutput(BaseModel):
    execution_summary: str
    test_results: List[TestExecutionResult]
    confirmed_vulnerabilities: List[ValidatedFinding]
    false_positives: List[ValidatedFinding]
    inconclusive_findings: List[ValidatedFinding] = Field(default_factory=list)
    reproducibility_notes: List[str] = Field(default_factory=list)


# Reporter Models
## Risk Statistics
class RiskClassification(BaseModel):
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    informational: int = 0


## Recommendation
class Recommendation(BaseModel):
    finding_id: str
    recommendation: str
    remediation_guidance: str
    mitigation_priority: Severity


## Audit Report
class AuditReport(BaseModel):
    project_name: str
    audit_scope: List[str]
    executive_summary: str
    risk_classification: RiskClassification
    findings: List[ValidatedFinding]
    recommendations: List[Recommendation]
    remediated_findings: List[str] = Field(default_factory=list)
    unresolved_risks: List[str] = Field(default_factory=list)
    conclusion: str


# Additional Important Infrastructure Models
## Agent Metadata: Useful for observability + debugging
class AgentExecutionMetadata(BaseModel):
    agent_name: str
    model_provider: str
    model_name: str
    temperature: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    execution_time_ms: Optional[int] = None


## Full Pipeline Artifact: Critical for replay/debugging
class AuditPipelineArtifact(BaseModel):
    strategist_output: StrategistOutput
    adversary_output: AdversaryOutput
    validator_output: ValidatorOutput
    final_report: AuditReport
    metadata: List[AgentExecutionMetadata]

