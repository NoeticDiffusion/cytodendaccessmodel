from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Payload = Any


@dataclass(slots=True)
class SpineState:
    """Local spine-level access state for a single dendritic branch.

    Spines represent the smallest unit of synaptic input and can have their own
    local access dynamics that couple with the branch-level state.

    Attributes:
        spine_id: Unique identifier for the spine.
        branch_id: Identifier of the parent branch.
        local_access: Multiplier for local input efficacy (0.0 to 1.0).
        coupling: Strength of coupling between spine and branch activation.
        local_drive: Instantaneous input drive to this specific spine.
        calcium_proxy: Phenomenological proxy for local calcium concentration.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    spine_id: str
    branch_id: str
    local_access: float = 1.0
    coupling: float = 1.0
    local_drive: float = 0.0
    calcium_proxy: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuralState:
    """Slow branch-level structural accessibility state.

    This represents the long-term, structural changes in the dendrite that
    modulate the fast activity. It typically corresponds to cytoskeletal
    stability or morphological changes.

    Attributes:
        branch_id: Identifier of the associated branch.
        accessibility: Current structural accessibility value (M_b).
        max_accessibility: Maximum possible structural accessibility.
        decay_rate: Passive decay rate of the structural state.
        noise_scale: Standard deviation of noise added during updates.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    branch_id: str
    accessibility: float = 0.5
    max_accessibility: float = 1.0
    decay_rate: float = 0.01
    noise_scale: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EligibilityTrace:
    """Local tag indicating recent branch-specific activity.

    Eligibility traces (E_b) mark branches that were recently active and
    are thus "eligible" for structural plasticity during consolidation.

    Attributes:
        branch_id: Identifier of the associated branch.
        value: Current magnitude of the eligibility trace.
        decay_rate: Rate at which the trace decays back to zero.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    branch_id: str
    value: float = 0.0
    decay_rate: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranslationReadiness:
    """Local capture or translation readiness used during consolidation.

    Translation readiness (P_b) represents the local availability of
    plasticity-related proteins or mRNA, triggered by replay or sleep.

    Attributes:
        branch_id: Identifier of the associated branch.
        value: Current magnitude of translation readiness.
        decay_rate: Rate at which readiness decays.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    branch_id: str
    value: float = 0.0
    decay_rate: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BranchState:
    """Fast dendritic state bundled with its slow branch-local variables.

    This is the primary unit of the simulation, representing a single
    dendritic branch with both its fast electrical/chemical state and
    its slow structural variables.

    Attributes:
        branch_id: Unique identifier for the branch.
        activation: Current activation level (effective_access * cue_drive).
        context_bias: Context-dependent modulatory input.
        inhibitory_tone: Local inhibitory input.
        fast_access: Transient accessibility driven by recent activity/context.
        slow_access: Persistent accessibility driven by structural state.
        effective_access: Combined accessibility (fast * slow).
        spines: Collection of spines attached to this branch.
        structural: Slow structural state for this branch.
        eligibility: Recent activity tag for plasticity.
        translation_readiness: Local protein/mRNA availability state.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    branch_id: str
    activation: float = 0.0
    context_bias: float = 0.0
    inhibitory_tone: float = 0.0
    fast_access: float = 0.0
    slow_access: float = 0.0
    effective_access: float = 0.0
    spines: tuple[SpineState, ...] = ()
    structural: StructuralState | None = None
    eligibility: EligibilityTrace | None = None
    translation_readiness: TranslationReadiness | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.structural is None:
            self.structural = StructuralState(branch_id=self.branch_id)
        if self.eligibility is None:
            self.eligibility = EligibilityTrace(branch_id=self.branch_id)
        if self.translation_readiness is None:
            self.translation_readiness = TranslationReadiness(branch_id=self.branch_id)


@dataclass(slots=True)
class TraceAllocation:
    """Branch-level allocation profile for a candidate engram trace.

    Defines how strongly a particular memory trace is mapped onto specific
    dendritic branches.

    Attributes:
        trace_id: Unique identifier for the memory trace.
        branch_weights: Mapping of branch IDs to their contribution weight.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    trace_id: str
    branch_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_branches(self, *, threshold: float = 0.0) -> tuple[str, ...]:
        """Returns branch IDs with weights above the specified threshold.

        Args:
            threshold: Minimum weight to include a branch.

        Returns:
            Tuple of branch IDs sorted by weight in descending order.
        """
        return tuple(
            branch_id
            for branch_id, weight in sorted(
                self.branch_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if weight > threshold
        )


@dataclass(slots=True)
class EngramTrace:
    """Memory trace defined by branch allocation.

    Unlike traditional vector-based memory, engram traces in this model are
    primarily defined by their distribution (allocation) across dendritic branches.

    Attributes:
        trace_id: Unique identifier for the engram.
        allocation: Branch-weight mapping defining the engram's physical footprint.
        label: Human-readable name for the engram.
        context: Context identifier this engram is associated with.
        salience: Relative importance or strength of the engram.
        replay_priority: Priority multiplier during consolidation replay.
        replay_count: Number of times this engram has been replayed.
        payload: Associated data or "content" of the memory trace.
        metadata: Arbitrary dictionary for additional state or tracking data.
    """

    trace_id: str
    allocation: TraceAllocation
    label: str | None = None
    context: str | None = None
    salience: float = 1.0
    replay_priority: float = 1.0
    replay_count: int = 0
    payload: Payload = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecallSupport:
    """Pre-threshold support for recall of one trace.

    Represents the internal "evidence" or "activation" for a specific memory
    trace before it passes through the final readout non-linearity.

    Attributes:
        trace_id: Identifier of the recalled engram.
        support: Raw cumulative activation across allocated branches.
        expressed_strength: Final output strength (usually 0.0 to 1.0).
        matched_context: Context string if it matched the current simulation context.
        active_branches: Branches contributing significantly to the recall.
        metadata: Additional diagnostic data about the recall event.
    """

    trace_id: str
    support: float = 0.0
    expressed_strength: float = 0.0
    matched_context: str | None = None
    active_branches: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsolidationWindow:
    """Replay or sleep-like window used to update slow structural state.

    Defines the parameters for an "offline" period where structural plasticity
    occurs based on recent activity (eligibility) and replay.

    Attributes:
        window_id: Unique identifier for the consolidation session.
        modulatory_drive: Global multiplier for structural learning (e.g., dopamine).
        sleep_drive: Global multiplier for sleep-specific translation (e.g., SWS).
        replay_trace_ids: Optional list of specific engrams to replay.
        context: Context during which consolidation occurs.
        metadata: Arbitrary dictionary for additional tracking data.
    """

    window_id: str = "offline"
    modulatory_drive: float = 1.0
    sleep_drive: float = 1.0
    replay_trace_ids: tuple[str, ...] = ()
    context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsolidationReport:
    """Summary of one consolidation pass.

    Provides statistics and diagnostics for a completed consolidation window.

    Attributes:
        window_id: Identifier of the consolidation window.
        branches_updated: Number of branches that underwent non-trivial changes.
        traces_replayed: List of engram IDs that were replayed.
        mean_structural_shift: Average change in accessibility (Delta M_b).
        mean_translation_readiness: Average P_b across all branches.
        metadata: Mirror of the window's parameters or additional stats.
    """

    window_id: str = "offline"
    branches_updated: int = 0
    traces_replayed: tuple[str, ...] = ()
    mean_structural_shift: float = 0.0
    mean_translation_readiness: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicsParameters:
    """Phenomenological parameters for the biological simulator scaffold.

    Governs the temporal and interaction dynamics of the fast (activity)
    and slow (structural) components of the model.

    Attributes:
        fast_gain: Gain on instantaneous cue-driven activation.
        context_gain: Gain on contextual modulation.
        inhibition_gain: Strength of local inhibition.
        spine_gain: Gain on spine-level local access contributions.
        structural_gain: Sensitivity of fast access to structural state.
        eligibility_decay: Decay rate of the activity-driven eligibility trace (E_b).
        translation_decay: Decay rate of the translation readiness state (P_b).
        structural_lr: Learning rate for structural accessibility (M_b).
        structural_decay: Passive decay of structural accessibility.
        structural_max: Ceiling for structural accessibility.
        readout_gain: Slope of the sigmoid readout function.
        readout_threshold: Inflection point of the sigmoid readout.
        replay_gain: Impact of replay events on translation readiness.
        sleep_gain: Multiplier for sleep-driven translation readiness.
        context_mismatch_penalty: Reduction in support when contexts don't match.
        structural_noise: Magnitude of random fluctuations in structural updates.
        translation_budget: Finite resource limit for total translation readiness.
        spillover_rate: Rate of neighboring branch influence during replay.
    """

    fast_gain: float = 2.0
    context_gain: float = 1.0
    inhibition_gain: float = 1.0
    spine_gain: float = 0.5
    structural_gain: float = 2.0
    eligibility_decay: float = 0.1
    translation_decay: float = 0.05
    structural_lr: float = 0.15
    structural_decay: float = 0.01
    structural_max: float = 1.0
    readout_gain: float = 5.0
    readout_threshold: float = 0.5
    replay_gain: float = 1.0
    sleep_gain: float = 1.0
    context_mismatch_penalty: float = 0.25
    structural_noise: float = 0.0
    translation_budget: float = 0.0
    spillover_rate: float = 0.0


@dataclass(slots=True)
class SimulationSnapshot:
    """Serializable snapshot of the simulator state at one step.

    Used for checkpointing, visualization, or diagnostic analysis.

    Attributes:
        step_index: Current iteration count of the simulator.
        context: The active context string at this step.
        branches: Full state of all dendritic branches.
        recall_supports: Latest computed support for all engram traces.
        metadata: Arbitrary dictionary for additional tracking data.
    """

    step_index: int = 0
    context: str | None = None
    branches: tuple[BranchState, ...] = ()
    recall_supports: tuple[RecallSupport, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

