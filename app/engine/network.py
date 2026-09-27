"""Entity networks built from trial records.

- source == target (e.g. intervention <-> intervention): co-occurrence. Two drugs are
  linked when a trial lists both; edge weight = number of such trials.
- source != target (e.g. sponsor <-> intervention): bipartite. A sponsor is linked to a
  drug when it leads a trial of that drug.

Pruning keeps the `top_n` busiest entities per side and the strongest edges among them,
and reports everything it dropped, so a readable graph never hides that it was trimmed.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from app.ctgov.extract import DimValue, TrialRow
from app.engine.aggregate import STUDY_URL
from app.schemas.enums import Dimension
from app.schemas.response import Citation, Edge, NetworkData, Node, Pruning

# Only these intervention types are "drugs" for network purposes.
DRUG_TYPES = {"DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"}
# Comparators that would otherwise become hubs linked to everything.
GENERIC_INTERVENTIONS = {
    "placebo",
    "placebos",
    "saline",
    "normal saline",
    "saline solution",
    "sodium chloride",
    "standard of care",
    "best supportive care",
    "vehicle",
}
EDGES_PER_NODE = 3


@dataclass
class NetworkResult:
    data: NetworkData
    pruning: Pruning
    excluded_generic: int  # trials' generic comparator mentions dropped


def _entities(row: TrialRow, dim: Dimension) -> tuple[list[DimValue], int]:
    values = list(row.get(dim))
    if dim != Dimension.INTERVENTION:
        return values, 0
    kept = [v for v in values if v.category in DRUG_TYPES and v.key not in GENERIC_INTERVENTIONS and "placebo" not in v.key]
    generic = sum(1 for v in values if v.key in GENERIC_INTERVENTIONS or "placebo" in v.key)
    return kept, generic


def _node_id(dim: Dimension, key: str) -> str:
    return f"{dim.value}:{key}"


def _citation(row: TrialRow, a: DimValue, b: DimValue | None = None) -> Citation:
    if b is None:
        return Citation(nct_id=row.nct_id, field=a.field, excerpt=a.excerpt, title=row.title, url=STUDY_URL.format(row.nct_id))
    field = a.field if a.field == b.field else f"{a.field} + {b.field}"
    return Citation(
        nct_id=row.nct_id,
        field=field,
        excerpt=f"{json.dumps(a.excerpt, ensure_ascii=False)} | {json.dumps(b.excerpt, ensure_ascii=False)}",
        title=row.title,
        url=STUDY_URL.format(row.nct_id),
    )


def build_network(rows: list[TrialRow], source: Dimension, target: Dimension, top_n: int, max_citations: int) -> NetworkResult:
    node_members: dict[str, list[tuple[TrialRow, DimValue]]] = defaultdict(list)
    node_meta: dict[str, tuple[str, str]] = {}  # id -> (label, type)
    edge_members: dict[tuple[str, str], list[tuple[TrialRow, DimValue, DimValue]]] = defaultdict(list)
    generic = 0

    for row in rows:
        left, g1 = _entities(row, source)
        generic += g1
        if source == target:
            pairs = combinations(sorted(left, key=lambda v: v.key), 2)
            right = left
        else:
            right, g2 = _entities(row, target)
            generic += g2
            pairs = ((a, b) for a in left for b in right)
        for dim, values in ((source, left), (target, right)):
            for v in values:
                nid = _node_id(dim, v.key)
                if not node_members[nid] or node_members[nid][-1][0] is not row:
                    node_members[nid].append((row, v))
                node_meta.setdefault(nid, (v.label, dim.value))
        for a, b in pairs:
            edge_members[(_node_id(source, a.key), _node_id(target, b.key))].append((row, a, b))

    # Keep the busiest entities on each side, then the strongest edges among them.
    by_type: dict[str, list[str]] = defaultdict(list)
    for nid in sorted(node_members, key=lambda n: (-len(node_members[n]), n)):
        by_type[node_meta[nid][1]].append(nid)
    keep = {nid for ids in by_type.values() for nid in ids[:top_n]}
    candidate_edges = sorted(
        (e for e in edge_members if e[0] in keep and e[1] in keep), key=lambda e: (-len(edge_members[e]), e)
    )
    max_edges = EDGES_PER_NODE * len(keep)
    kept_edges = candidate_edges[:max_edges]
    min_weight = min((len(edge_members[e]) for e in kept_edges), default=None)
    connected = {n for e in kept_edges for n in e}

    nodes = [
        Node(
            id=nid,
            label=node_meta[nid][0],
            type=node_meta[nid][1],
            size=len(node_members[nid]),
            supporting_nct_ids=[r.nct_id for r, _ in node_members[nid]],
            citation_count=len(node_members[nid]),
            citations=[_citation(r, v) for r, v in node_members[nid][:max_citations]],
        )
        for nid in sorted(connected, key=lambda n: (-len(node_members[n]), n))
    ]
    edges = [
        Edge(
            source=s,
            target=t,
            weight=len(edge_members[(s, t)]),
            supporting_nct_ids=[r.nct_id for r, _, _ in edge_members[(s, t)]],
            citation_count=len(edge_members[(s, t)]),
            citations=[_citation(r, a, b) for r, a, b in edge_members[(s, t)][:max_citations]],
        )
        for s, t in kept_edges
    ]
    pruning = Pruning(
        top_n=top_n,
        min_edge_weight=min_weight,
        nodes_dropped=len(node_members) - len(nodes),
        edges_dropped=len(edge_members) - len(edges),
    )
    return NetworkResult(NetworkData(nodes=nodes, edges=edges), pruning, generic)
