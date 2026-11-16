"""Biological spike, LIF motor-neuron, and MaleCNS connectome interfaces for v2."""

from .lif_interface import (
    LIFBiophysicsParams,
    LIFDriveResult,
    analytical_lif_current_mv,
    optimize_premotor_connectome_drive,
    realize_direct_lif_mn_drive,
    simulate_lif_substep_trace,
)
from .malecns_pools import (
    LF_15_DISJOINT_POOLS,
    PremotorConnectome,
    load_malecns_premotor_subnetwork,
    resolve_motor_pools,
)
from .spiking import (
    SpikeRealizationResult,
    deconvolve_target_excitation_to_rates,
    optimize_rate_spikes_with_physics_vjp,
    realize_rate_spikes,
)

__all__ = [
    "LF_15_DISJOINT_POOLS",
    "LIFBiophysicsParams",
    "LIFDriveResult",
    "PremotorConnectome",
    "SpikeRealizationResult",
    "analytical_lif_current_mv",
    "deconvolve_target_excitation_to_rates",
    "load_malecns_premotor_subnetwork",
    "optimize_premotor_connectome_drive",
    "optimize_rate_spikes_with_physics_vjp",
    "realize_direct_lif_mn_drive",
    "realize_rate_spikes",
    "resolve_motor_pools",
    "simulate_lif_substep_trace",
]
