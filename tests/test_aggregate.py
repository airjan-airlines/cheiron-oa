from app.ctgov.extract import extract
from app.engine.aggregate import evidence, group, overlap_note
from app.schemas.enums import Dimension as D
from tests.helpers import fetched, trial


def test_phase_buckets_partition_the_trials_exactly():
    q = fetched("korea_gastric_recruiting")
    g = group(q.rows, D.PHASE, top_n=15)
    assert sum(b.count for b in g.buckets) == len(q.rows) == q.total_matching
    assert g.unclassified_count == 0


def test_phase_buckets_follow_clinical_order_not_count():
    rows = [extract(trial(i, designModule={"phases": p})) for i, p in enumerate([["PHASE3"], ["PHASE3"], ["PHASE1"], ["NA"]])]
    rows.append(extract(trial(9)))  # no phase at all
    assert [b.key for b in group(rows, D.PHASE, top_n=15).buckets] == ["PHASE1", "PHASE3", "NA", "MISSING"]


def test_count_ordered_dimensions_are_cut_to_top_n_and_report_the_rest():
    q = fetched("tirzepatide")
    g = group(q.rows, D.COUNTRY, top_n=5)
    counts = [b.count for b in g.buckets]
    assert len(counts) == 5 and counts == sorted(counts, reverse=True)
    assert g.not_shown > 0


def test_years_are_zero_filled_within_filter_bounds():
    rows = [extract(trial(1, statusModule={"startDateStruct": {"date": "2016-03"}})), extract(trial(2, statusModule={"startDateStruct": {"date": "2019"}}))]
    g = group(rows, D.START_YEAR, top_n=15, year_range=(2015, 2020))
    assert [(b.key, b.count) for b in g.buckets] == [("2015", 0), ("2016", 1), ("2017", 0), ("2018", 0), ("2019", 1), ("2020", 0)]


def test_trials_without_a_value_are_unclassified_not_dropped_silently():
    rows = [extract(trial(1, statusModule={"startDateStruct": {"date": "2016"}})), extract(trial(2))]
    g = group(rows, D.START_YEAR, top_n=15)
    assert g.unclassified_count == 1
    assert [b.key for b in g.buckets] == ["2016"]


def test_evidence_lists_every_trial_but_caps_excerpts():
    q = fetched("korea_gastric_recruiting")
    bucket = max(group(q.rows, D.PHASE, top_n=15).buckets, key=lambda b: b.count)
    ev = evidence(bucket.members, max_citations=2)
    assert len(ev["supporting_nct_ids"]) == ev["citation_count"] == bucket.count
    assert len(ev["citations"]) == 2
    first_row, first_value = bucket.members[0]
    assert ev["citations"][0].excerpt == first_value.excerpt
    assert ev["citations"][0].url == f"https://clinicaltrials.gov/study/{first_row.nct_id}"


def test_overlap_note_only_for_overlapping_dimensions():
    q = fetched("tirzepatide")
    assert overlap_note(q.rows, D.PHASE) is None
    assert "memberships" in overlap_note(q.rows, D.COUNTRY)
