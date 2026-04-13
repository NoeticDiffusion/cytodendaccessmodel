"""
Cytodendritic Accessibility Model (CytoDend).

This package provides a biologically constrained simulation scaffold for
modelling associative memory through dendritic branch-level structural
accessibility and fast activity dynamics.
"""

from cytodend_accessmodel.contracts import (
    BranchState,
    ConsolidationReport,
    ConsolidationWindow,
    DynamicsParameters,
    EligibilityTrace,
    EngramTrace,
    RecallSupport,
    SimulationSnapshot,
    SpineState,
    StructuralState,
    TraceAllocation,
    TranslationReadiness,
)
from cytodend_accessmodel.simulator import CytodendAccessModelSimulator

__all__ = [
    "BranchState",
    "ConsolidationReport",
    "ConsolidationWindow",
    "CytodendAccessModelSimulator",
    "DynamicsParameters",
    "EligibilityTrace",
    "EngramTrace",
    "RecallSupport",
    "SimulationSnapshot",
    "SpineState",
    "StructuralState",
    "TraceAllocation",
    "TranslationReadiness",
]

