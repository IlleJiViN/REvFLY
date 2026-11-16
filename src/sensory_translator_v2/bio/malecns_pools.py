"""MaleCNS motor-neuron pool metadata and premotor connectome access for Sensory Translator v2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import AssetValidationError, LF_ACTUATOR_NAMES
from ..rate.bridge import LF_15_POOLS, LF_TIBIA_POOLS, MotorPool

_MALECNS_NEURONS = Path("external/flyboard/data/neurons.parquet")
_MALECNS_GRAPH = Path("external/flyboard/data/graph_min5.npz")

# Disjoint sub-branch allocation across the 39 unique MaleCNS LF motor neuron bodyIds
# where multi-branch actuators in configs/neuromuscular/mapping_evidence.json share candidate pools.
LF_15_DISJOINT_POOLS: tuple[MotorPool, ...] = (
    MotorPool("LFC_tergopleural_promotor_a", (800061, 801173)),
    MotorPool("LFC_tergopleural_promotor_b", (807281,)),
    MotorPool("LFC_pleural_remotor_and_abductor", (802509, 803803)),
    MotorPool("LFC_pleural_promotor", (934091,)),
    MotorPool("LFC_sternal_anterior_rotator", (806923, 919509)),
    MotorPool("LFC_sternal_posterior_rotator", (801915, 804851, 808640, 1050335829)),
    MotorPool("LFC_sternal_adductor", (813480,)),
    MotorPool("LFF_trochanter_flexor_b", (810639, 814489, 818389, 823168)),
    MotorPool("LFF_sterno-tergo-trochanter_extensor_a", (801685, 802361, 802552)),
    MotorPool("LFF_sterno-tergo-trochanter_extensor_b", (803237, 804465, 805741)),
    MotorPool("LFF_accesory_trochanter_flexor", (821291, 909618, 911134)),
    MotorPool("LFF_trochanter_extensor", (803412, 807906)),
    MotorPool("LFF_trochanter_flexor_a", (823281, 828939, 1051367796, 1051670893)),
    MotorPool("LFTibia_flex_93434", (807165, 809912, 818057, 819384, 909831)),
    MotorPool("LFTibia_extensor_93932", (800636, 815344)),
)


@dataclass(frozen=True)
class PremotorConnectome:
    """Extracted MaleCNS premotor -> LF motor-neuron subnetwork from neurons.parquet + graph_min5.npz."""

    scope: str
    pools: tuple[MotorPool, ...]
    mn_body_ids: tuple[int, ...]
    mn_types: tuple[str, ...]
    pre_body_ids: np.ndarray
    pre_types: tuple[str, ...]
    pre_superclasses: tuple[str, ...]
    pre_neurotransmitters: tuple[str, ...]
    weight_pre_to_mn: np.ndarray  # [N_mn, N_pre], signed MaleCNS synapse counts
    weight_mn_to_mn: np.ndarray   # [N_mn, N_mn], signed intra-MN synapses
    edge_count: int
    positive_edge_count: int
    negative_edge_count: int
    partitioned_shared_subbranches: bool

    def summary(self) -> dict[str, Any]:
        sc_counts: dict[str, int] = {}
        for sc in self.pre_superclasses:
            sc_counts[sc] = sc_counts.get(sc, 0) + 1
        nt_counts: dict[str, int] = {}
        for nt in self.pre_neurotransmitters:
            nt_counts[nt] = nt_counts.get(nt, 0) + 1
        return {
            "scope": self.scope,
            "num_pools": len(self.pools),
            "num_motor_neurons": len(self.mn_body_ids),
            "num_premotor_neurons": int(self.pre_body_ids.shape[0]),
            "edge_count": self.edge_count,
            "positive_edge_count": self.positive_edge_count,
            "negative_edge_count": self.negative_edge_count,
            "partitioned_shared_subbranches": self.partitioned_shared_subbranches,
            "premotor_superclasses": sc_counts,
            "premotor_neurotransmitters": nt_counts,
        }


def resolve_motor_pools(
    scope: str = "lf15",
    *,
    disjoint_subbranches: bool = True,
    mapping_path: str | Path | None = None,
) -> tuple[MotorPool, ...]:
    """Return ordered MotorPool tuple for 'tibia' (2 pools) or 'lf15' (15 actuators)."""
    if scope == "tibia":
        return LF_TIBIA_POOLS
    if scope != "lf15":
        raise ValueError(f"scope must be 'tibia' or 'lf15', got {scope!r}")
    if disjoint_subbranches:
        return LF_15_DISJOINT_POOLS
    if mapping_path is not None and Path(mapping_path).is_file():
        payload = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        by_act = {
            str(row["actuator"]): tuple(int(b) for b in row["bodyIds"])
            for row in payload["complete_actuator_to_motor_pool_table"]
        }
        return tuple(MotorPool(name=act, body_ids=by_act[act]) for act in LF_ACTUATOR_NAMES)
    return LF_15_POOLS


def premotor_unavailability(asset_root: Path) -> str | None:
    """Describe missing local inputs to the premotor connectome path."""
    missing = [path.name for path in (_MALECNS_NEURONS, _MALECNS_GRAPH) if not (asset_root / path).is_file()]
    if missing:
        return f"Missing local MaleCNS data: {', '.join(missing)}"
    if find_spec("pandas") is None or not (find_spec("pyarrow") or find_spec("fastparquet")):
        return "Reading MaleCNS data requires pandas and a Parquet reader (pyarrow or fastparquet)"
    return None


def load_malecns_premotor_subnetwork(
    repo_root: str | Path,
    *,
    scope: str = "tibia",
    disjoint_subbranches: bool = True,
    min_abs_synapse_weight: float = 5.0,
) -> PremotorConnectome:
    """Extract exact presynaptic MaleCNS premotor connectome targeting LF motor neurons."""
    root = Path(repo_root).expanduser().resolve()
    mapping_file = root / "configs/neuromuscular/mapping_evidence.json"
    neurons_file = root / _MALECNS_NEURONS
    graph_file = root / _MALECNS_GRAPH
    for p in (neurons_file, graph_file):
        if not p.is_file():
            raise AssetValidationError(f"Required MaleCNS data file missing: {p}")

    import pandas as pd

    pools = resolve_motor_pools(
        scope,
        disjoint_subbranches=disjoint_subbranches,
        mapping_path=mapping_file if mapping_file.is_file() else None,
    )
    mn_body_ids = tuple(sorted({bid for pool in pools for bid in pool.body_ids}))

    neurons_df = pd.read_parquet(
        neurons_file,
        columns=["bodyId", "type", "superclass", "consensus_nt"],
    )
    body_ids_all = neurons_df["bodyId"].to_numpy(dtype=np.int64)
    bid_to_gidx = {int(bid): idx for idx, bid in enumerate(body_ids_all)}
    missing_mns = [bid for bid in mn_body_ids if bid not in bid_to_gidx]
    if missing_mns:
        raise AssetValidationError(f"LF motor neuron bodyIds missing in neurons.parquet: {missing_mns}")

    mn_gidxs = np.array([bid_to_gidx[bid] for bid in mn_body_ids], dtype=np.int32)
    mn_gidx_to_row = {int(gidx): r for r, gidx in enumerate(mn_gidxs)}

    with np.load(graph_file, allow_pickle=False) as graph:
        pre = np.asarray(graph["pre"], dtype=np.int32)
        post = np.asarray(graph["post"], dtype=np.int32)
        w = np.asarray(graph["w"], dtype=np.float64)

    edge_mask = np.isin(post, mn_gidxs) & (np.abs(w) >= float(min_abs_synapse_weight))
    sel_pre = pre[edge_mask]
    sel_post = post[edge_mask]
    sel_w = w[edge_mask]

    mn_gidx_set = set(int(g) for g in mn_gidxs)
    unique_pre_gidxs = np.array(
        sorted({int(g) for g in sel_pre if int(g) not in mn_gidx_set}),
        dtype=np.int32,
    )
    pre_gidx_to_col = {int(g): c for c, g in enumerate(unique_pre_gidxs)}

    w_pre_mn = np.zeros((len(mn_body_ids), len(unique_pre_gidxs)), dtype=np.float64)
    w_mn_mn = np.zeros((len(mn_body_ids), len(mn_body_ids)), dtype=np.float64)
    pos_edges = 0
    neg_edges = 0
    for p_g, q_g, w_val in zip(sel_pre, sel_post, sel_w):
        row = mn_gidx_to_row[int(q_g)]
        p_int = int(p_g)
        if p_int in mn_gidx_to_row:
            w_mn_mn[row, mn_gidx_to_row[p_int]] += float(w_val)
        else:
            col = pre_gidx_to_col[p_int]
            w_pre_mn[row, col] += float(w_val)
        if w_val > 0.0:
            pos_edges += 1
        elif w_val < 0.0:
            neg_edges += 1

    types_arr = neurons_df["type"].fillna("unknown").astype(str).to_numpy()
    sc_arr = neurons_df["superclass"].fillna("unknown").astype(str).to_numpy()
    nt_arr = neurons_df["consensus_nt"].fillna("unclear").astype(str).to_numpy()

    return PremotorConnectome(
        scope=scope,
        pools=pools,
        mn_body_ids=mn_body_ids,
        mn_types=tuple(str(types_arr[g]) for g in mn_gidxs),
        pre_body_ids=body_ids_all[unique_pre_gidxs],
        pre_types=tuple(str(types_arr[g]) for g in unique_pre_gidxs),
        pre_superclasses=tuple(str(sc_arr[g]) for g in unique_pre_gidxs),
        pre_neurotransmitters=tuple(str(nt_arr[g]) for g in unique_pre_gidxs),
        weight_pre_to_mn=w_pre_mn,
        weight_mn_to_mn=w_mn_mn,
        edge_count=int(np.count_nonzero(edge_mask)),
        positive_edge_count=pos_edges,
        negative_edge_count=neg_edges,
        partitioned_shared_subbranches=bool(scope == "lf15" and disjoint_subbranches),
    )
