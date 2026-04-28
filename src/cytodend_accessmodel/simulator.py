from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import exp
from random import gauss
from typing import Iterable, Mapping

from cytodend_accessmodel.contracts import (
    BranchState,
    ConsolidationReport,
    ConsolidationWindow,
    DynamicsParameters,
    EngramTrace,
    RecallSupport,
    SimulationSnapshot,
    SpineState,
    StructuralState,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = exp(value)
    return exp_term / (1.0 + exp_term)


@dataclass(slots=True)
class CytodendAccessModelSimulator:
    """Biologically constrained branch-and-structure simulator scaffold.

    The simulator manages the lifecycle and interaction dynamics of dendritic
    branches, memory traces (engrams), and slow structural plasticity.
    It follows a two-pass architecture for consolidation and a sigmoid-driven
    fast access/readout mechanism.

    Attributes:
        branches: Mapping of branch IDs to their current state.
        traces: Mapping of trace IDs to engram definitions.
        parameters: Global dynamical parameters for the simulator.
        context: Current environmental or modulatory context.
        step_index: Sequential counter for cue application steps.
        last_recall_supports: Latest computed support values for all engrams.
        branch_adjacency: Neighbor connectivity used for spillover effects.
    """

    branches: dict[str, BranchState]
    traces: dict[str, EngramTrace] = field(default_factory=dict)
    parameters: DynamicsParameters = field(default_factory=DynamicsParameters)
    context: str | None = None
    step_index: int = 0
    last_recall_supports: tuple[RecallSupport, ...] = ()
    branch_adjacency: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_branch_ids(
        cls,
        branch_ids: Iterable[str],
        *,
        spines_per_branch: int = 0,
        parameters: DynamicsParameters | None = None,
    ) -> "CytodendAccessModelSimulator":
        """Factory method to initialize the simulator with a set of branch IDs.

        Args:
            branch_ids: Collection of unique IDs for the branches.
            spines_per_branch: Number of spines to initialize on each branch.
            parameters: Optional custom dynamics parameters.

        Returns:
            A new instance of the simulator.
        """
        effective_params = parameters or DynamicsParameters()
        branches: dict[str, BranchState] = {}
        for branch_id in branch_ids:
            spines = tuple(
                SpineState(
                    spine_id=f"{branch_id}:spine:{idx}",
                    branch_id=branch_id,
                )
                for idx in range(spines_per_branch)
            )
            structural = StructuralState(
                branch_id=branch_id,
                decay_rate=effective_params.structural_decay,
                max_accessibility=effective_params.structural_max,
                noise_scale=effective_params.structural_noise,
            )
            branches[branch_id] = BranchState(
                branch_id=branch_id,
                spines=spines,
                structural=structural,
            )
        return cls(
            branches=branches,
            parameters=effective_params,
        )

    def add_trace(self, trace: EngramTrace) -> None:
        """Adds a memory trace (engram) to the simulator's registry.

        Args:
            trace: The engram trace definition.
        """
        self.traces[trace.trace_id] = trace

    def set_context(self, context: str | None) -> None:
        """Sets the active modulatory or environmental context.

        Args:
            context: Identifier for the context (None for neutral/default).
        """
        self.context = context

    def apply_cue(
        self,
        cue_inputs: Mapping[str, float],
        *,
        context: str | None = None,
        context_bias: Mapping[str, float] | None = None,
        inhibitory_tone: Mapping[str, float] | None = None,
    ) -> tuple[RecallSupport, ...]:
        """Applies a set of cue inputs and updates the fast state of all branches.

        This represents one "online" step where cues drive dendritic activation,
        modulated by context, local access, and inhibition.

        Args:
            cue_inputs: Mapping of branch IDs to raw cue drive magnitude.
            context: Optional context switch to apply before processing.
            context_bias: Additional modulatory bias per branch.
            inhibitory_tone: Local inhibitory drive per branch.

        Returns:
            Computed recall supports for all registered engrams.
        """
        if context is not None:
            self.context = context

        context_bias = context_bias or {}
        inhibitory_tone = inhibitory_tone or {}

        for branch in self.branches.values():
            cue_drive = float(cue_inputs.get(branch.branch_id, 0.0))
            branch.context_bias = float(context_bias.get(branch.branch_id, 0.0))
            branch.inhibitory_tone = float(inhibitory_tone.get(branch.branch_id, 0.0))

            # Aggregate spine contributions: local spine access gates branch drive.
            spine_support = self._aggregate_spine_access(branch)

            # Fast accessibility term combines cue, context, spines, and inhibition.
            fast_term = (
                self.parameters.fast_gain * cue_drive
                + self.parameters.context_gain * branch.context_bias
                + self.parameters.spine_gain * spine_support
                - self.parameters.inhibition_gain * branch.inhibitory_tone
            )
            branch.fast_access = _sigmoid(fast_term)

            # Slow accessibility term is derived from the long-term structural state M_b.
            branch.slow_access = _sigmoid(
                self.parameters.structural_gain * branch.structural.accessibility
            )

            # Effective accessibility gates the final activation of the branch.
            branch.effective_access = _clamp01(branch.fast_access * branch.slow_access)
            branch.activation = branch.effective_access * cue_drive

            # Eligibility trace (E_b) marks the branch for future structural updates.
            branch.eligibility.value = _clamp01(
                (1.0 - self.parameters.eligibility_decay) * branch.eligibility.value
                + abs(branch.activation)
            )

            # Passive decay of translation readiness (P_b) between consolidation events.
            branch.translation_readiness.value = _clamp01(
                (1.0 - self.parameters.translation_decay)
                * branch.translation_readiness.value
            )

            for spine in branch.spines:
                spine.local_drive = cue_drive * spine.local_access
                spine.calcium_proxy = abs(branch.activation) * spine.coupling

        self.step_index += 1
        self.last_recall_supports = self.compute_recall_supports()
        return self.last_recall_supports

    def compute_recall_supports(self) -> tuple[RecallSupport, ...]:
        """Calculates recall support for all engrams based on current activation.

        Returns:
            Sorted tuple of RecallSupport objects (highest support first).
        """
        supports: list[RecallSupport] = []

        for trace in self.traces.values():
            support = 0.0
            active_branches: list[str] = []

            for branch_id, weight in trace.allocation.branch_weights.items():
                branch = self.branches.get(branch_id)
                if branch is None:
                    continue
                contribution = float(weight) * branch.activation
                support += contribution
                if branch.effective_access > 0.5 and branch.activation > 0.0:
                    active_branches.append(branch_id)

            if self.context is not None and trace.context is not None and trace.context != self.context:
                support *= max(0.0, 1.0 - self.parameters.context_mismatch_penalty)

            expressed_strength = _sigmoid(
                self.parameters.readout_gain
                * (support - self.parameters.readout_threshold)
            )
            supports.append(
                RecallSupport(
                    trace_id=trace.trace_id,
                    support=support,
                    expressed_strength=expressed_strength,
                    matched_context=trace.context if trace.context == self.context else None,
                    active_branches=tuple(active_branches),
                    metadata={
                        "label": trace.label,
                        "replay_priority": trace.replay_priority,
                    },
                )
            )

        supports.sort(key=lambda item: item.support, reverse=True)
        return tuple(supports)

    def run_consolidation(
        self,
        window: ConsolidationWindow | None = None,
    ) -> ConsolidationReport:
        """Runs the "offline" consolidation process to update slow structural state.

        This process follows two main passes:

        - Update translation readiness (P_b) based on replay and sleep.
        - Update structural accessibility (M_b) based on eligibility (E_b)
          and readiness (P_b).

        Args:
            window: Parameters defining the consolidation period.

        Returns:
            A report containing summary statistics of the updates.
        """
        window = window or ConsolidationWindow()
        replay_ids = self._select_replay_ids(window)

        # Pre-compute replay overlap once per branch for both passes.
        # Overlap represents how strongly the replayed traces cover each branch.
        replay_overlaps: dict[str, float] = {
            bid: self._replay_overlap(bid, replay_ids)
            for bid in self.branches
        }

        # --- Pass 1: update P_b (translation readiness) for all branches ---
        # P_b tracks local protein/mRNA availability triggered by replay/sleep.
        for branch in self.branches.values():
            overlap = replay_overlaps[branch.branch_id]
            branch.translation_readiness.value = _clamp01(
                (1.0 - self.parameters.translation_decay)
                * branch.translation_readiness.value
                + self.parameters.replay_gain * overlap
                + self.parameters.sleep_gain * window.sleep_drive * overlap
            )

        # --- Budget normalization: finite translation resource competition ---
        # Models the finite availability of plasticity-related proteins.
        if self.parameters.translation_budget > 0.0:
            total_p = sum(
                b.translation_readiness.value for b in self.branches.values()
            )
            if total_p > self.parameters.translation_budget:
                scale = self.parameters.translation_budget / total_p
                for b in self.branches.values():
                    b.translation_readiness.value = _clamp01(
                        b.translation_readiness.value * scale
                    )

        # --- Pass 2: update M_b using (normalized) P_b, then decay E_b ---
        # M_b represents slow structural accessibility (e.g., cytoskeletal stability).
        structural_shifts: list[float] = []
        readiness_values: list[float] = []
        branches_updated = 0

        for branch in self.branches.values():
            old_accessibility = branch.structural.accessibility
            m_max = branch.structural.max_accessibility

            # Structural update Delta M_b: driven by eligibility (E_b),
            # translation readiness (P_b), and modulatory drive.
            structural_delta = (
                self.parameters.structural_lr
                * branch.eligibility.value
                * branch.translation_readiness.value
                * max(0.0, window.modulatory_drive)
                * (1.0 - old_accessibility / m_max)
                - branch.structural.decay_rate * old_accessibility
            )

            # Apply optional stochasticity to the structural update.
            noise_term = (
                gauss(0.0, branch.structural.noise_scale)
                if branch.structural.noise_scale > 0.0
                else 0.0
            )

            branch.structural.accessibility = _clamp(
                old_accessibility + structural_delta + noise_term,
                0.0,
                m_max,
            )

            # Update slow access cache based on the new structural state.
            branch.slow_access = _sigmoid(
                self.parameters.structural_gain * branch.structural.accessibility
            )
            branch.effective_access = _clamp01(branch.fast_access * branch.slow_access)

            # Tag decay: eligibility trace decays after it is "used" in the update.
            branch.eligibility.value = _clamp01(
                (1.0 - self.parameters.eligibility_decay) * branch.eligibility.value
            )

            shift = branch.structural.accessibility - old_accessibility
            structural_shifts.append(shift)
            readiness_values.append(branch.translation_readiness.value)
            if abs(shift) > 1e-12:
                branches_updated += 1

        for trace_id in replay_ids:
            trace = self.traces.get(trace_id)
            if trace is not None:
                trace.replay_count += 1

        mean_structural_shift = (
            sum(structural_shifts) / len(structural_shifts) if structural_shifts else 0.0
        )
        mean_translation_readiness = (
            sum(readiness_values) / len(readiness_values) if readiness_values else 0.0
        )

        return ConsolidationReport(
            window_id=window.window_id,
            branches_updated=branches_updated,
            traces_replayed=tuple(replay_ids),
            mean_structural_shift=mean_structural_shift,
            mean_translation_readiness=mean_translation_readiness,
            metadata={
                "context": window.context,
                "sleep_drive": window.sleep_drive,
                "modulatory_drive": window.modulatory_drive,
            },
        )

    def snapshot(self) -> SimulationSnapshot:
        """Captures a serializable snapshot of the current simulator state.

        Returns:
            A SimulationSnapshot instance.
        """
        return SimulationSnapshot(
            step_index=self.step_index,
            context=self.context,
            branches=tuple(deepcopy(branch) for branch in self.branches.values()),
            recall_supports=tuple(deepcopy(item) for item in self.last_recall_supports),
        )

    def _aggregate_spine_access(self, branch: BranchState) -> float:
        """Aggregates local access contributions from all spines on a branch.

        Args:
            branch: The branch to aggregate from.

        Returns:
            Normalized aggregate spine access (1.0 if no spines).
        """
        if not branch.spines:
            return 1.0
        return sum(spine.local_access * spine.coupling for spine in branch.spines) / len(
            branch.spines
        )

    def _select_replay_ids(self, window: ConsolidationWindow) -> tuple[str, ...]:
        """Determines which traces will be replayed during a consolidation window.

        Args:
            window: Parameters defining the consolidation.

        Returns:
            Tuple of trace IDs to replay.
        """
        if window.replay_trace_ids:
            return tuple(
                trace_id for trace_id in window.replay_trace_ids if trace_id in self.traces
            )

        selected_ids: list[str] = []
        for trace in self.traces.values():
            if window.context is not None and trace.context not in (None, window.context):
                continue
            selected_ids.append(trace.trace_id)
        return tuple(selected_ids)

    def _direct_overlap(self, branch_id: str, replay_ids: tuple[str, ...]) -> float:
        """Calculates direct replay overlap for a branch without neighborhood spillover.

        Args:
            branch_id: Target branch ID.
            replay_ids: Set of traces being replayed.

        Returns:
            Normalized weighted overlap (0.0 to 1.0).
        """
        if not replay_ids:
            return 0.0
        weighted_overlap = 0.0
        total_priority = 0.0
        for trace_id in replay_ids:
            trace = self.traces.get(trace_id)
            if trace is None:
                continue
            priority = max(0.0, trace.replay_priority)
            weighted_overlap += priority * trace.allocation.branch_weights.get(branch_id, 0.0)
            total_priority += priority
        if total_priority == 0.0:
            return 0.0
        return weighted_overlap / total_priority

    def _replay_overlap(self, branch_id: str, replay_ids: tuple[str, ...]) -> float:
        """Calculates effective replay overlap including neighborhood spillover.

        Args:
            branch_id: Target branch ID.
            replay_ids: Set of traces being replayed.

        Returns:
            Effective overlap value (0.0 to 1.0).
        """
        direct = self._direct_overlap(branch_id, replay_ids)
        if self.parameters.spillover_rate <= 0.0 or not self.branch_adjacency:
            return direct
        neighbors = self.branch_adjacency.get(branch_id, ())
        if not neighbors:
            return direct
        neighbor_overlaps = [
            self._direct_overlap(nid, replay_ids)
            for nid in neighbors
            if nid in self.branches
        ]
        if not neighbor_overlaps:
            return direct
        spillover = self.parameters.spillover_rate * (
            sum(neighbor_overlaps) / len(neighbor_overlaps)
        )
        return _clamp01(direct + spillover)

