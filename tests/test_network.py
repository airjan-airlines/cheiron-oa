import json

from app.ctgov.extract import extract
from app.engine.network import build_network
from app.schemas.enums import Dimension as D
from tests.helpers import fetched, trial


def _drug_trial(nct: int, sponsor: str, *drugs: str) -> dict:
    return trial(
        nct,
        sponsorCollaboratorsModule={"leadSponsor": {"name": sponsor, "class": "INDUSTRY"}},
        armsInterventionsModule={"interventions": [{"type": "DRUG", "name": d} for d in drugs]},
    )


def test_co_occurrence_edges_count_trials_listing_both_drugs():
    rows = [
        extract(_drug_trial(1, "S", "Carboplatin", "Pemetrexed")),
        extract(_drug_trial(2, "S", "Carboplatin", "Pemetrexed", "Placebo")),
        extract(_drug_trial(3, "S", "Carboplatin")),
    ]
    net = build_network(rows, D.INTERVENTION, D.INTERVENTION, top_n=10, max_citations=5)
    (edge,) = net.data.edges
    assert (edge.source, edge.target, edge.weight) == ("intervention:carboplatin", "intervention:pemetrexed", 2)
    assert edge.supporting_nct_ids == ["NCT00000001", "NCT00000002"]
    assert [json.loads(p) for p in edge.citations[0].excerpt.split(" | ")] == ["Carboplatin", "Pemetrexed"]
    sizes = {n.id: n.size for n in net.data.nodes}
    assert sizes == {"intervention:carboplatin": 3, "intervention:pemetrexed": 2}
    assert net.excluded_generic == 1  # placebo left out rather than becoming a hub


def test_bipartite_network_links_sponsors_to_their_drugs():
    rows = [extract(_drug_trial(1, "Merck", "Pembrolizumab")), extract(_drug_trial(2, "Merck", "Pembrolizumab")), extract(_drug_trial(3, "Lilly", "Tirzepatide"))]
    net = build_network(rows, D.SPONSOR, D.INTERVENTION, top_n=10, max_citations=1)
    weights = {(e.source, e.target): e.weight for e in net.data.edges}
    assert weights == {("sponsor:merck", "intervention:pembrolizumab"): 2, ("sponsor:lilly", "intervention:tirzepatide"): 1}
    assert {n.type for n in net.data.nodes} == {"sponsor", "intervention"}


def test_pruning_is_reported_and_edges_only_reference_kept_nodes():
    q = fetched("kras_nsclc")
    net = build_network(q.rows, D.INTERVENTION, D.INTERVENTION, top_n=5, max_citations=1)
    ids = {n.id for n in net.data.nodes}
    assert len(ids) <= 5
    assert all(e.source in ids and e.target in ids for e in net.data.edges)
    assert net.pruning.nodes_dropped > 0 and net.pruning.edges_dropped > 0
    assert net.pruning.min_edge_weight == min(e.weight for e in net.data.edges)
