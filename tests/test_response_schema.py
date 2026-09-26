import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.response import (
    BarChart,
    Channel,
    Citation,
    DataRow,
    NetworkData,
    Visualization,
    VisualizeResponse,
    XYEncoding,
)

PHASE_ENCODING = XYEncoding(
    x=Channel(field="phase", type="ordinal", label="Phase", label_field="phase_label"),
    y=Channel(field="trial_count", type="quantitative", label="Trials"),
)


def _phase_row(**overrides):
    fields = {"phase": "PHASE3", "phase_label": "Phase 3", "trial_count": 2, "supporting_nct_ids": ["NCT00000001", "NCT00000002"], "citation_count": 2}
    fields.update(overrides)
    return DataRow(**fields)


def test_bar_chart_accepts_rows_with_all_encoded_fields():
    chart = BarChart(title="Trials by phase", encoding=PHASE_ENCODING, data=[_phase_row()])
    dumped = chart.model_dump(mode="json")
    assert dumped["type"] == "bar_chart"
    assert dumped["data"][0]["phase_label"] == "Phase 3"
    assert dumped["data"][0]["supporting_nct_ids"] == ["NCT00000001", "NCT00000002"]


def test_rows_missing_an_encoded_field_are_rejected():
    bad = DataRow(phase="PHASE3", trial_count=2)  # no phase_label
    with pytest.raises(ValidationError, match="missing encoded field 'phase_label'"):
        BarChart(title="t", encoding=PHASE_ENCODING, data=[bad])


def test_visualization_union_dispatches_on_type():
    payload = {
        "type": "bar_chart",
        "title": "t",
        "encoding": PHASE_ENCODING.model_dump(),
        "data": [_phase_row().model_dump()],
    }
    viz = TypeAdapter(Visualization).validate_python(payload)
    assert isinstance(viz, BarChart)


def test_network_edges_must_reference_existing_nodes():
    with pytest.raises(ValidationError, match="missing node"):
        NetworkData(
            nodes=[{"id": "a", "label": "A", "type": "sponsor", "size": 1}],
            edges=[{"source": "a", "target": "b", "weight": 1}],
        )


def test_citation_requires_a_real_nct_id():
    with pytest.raises(ValidationError):
        Citation(nct_id="12345", field="f", excerpt="x", url="u")


def test_ok_status_requires_a_visualization_and_others_forbid_it():
    with pytest.raises(ValidationError):
        VisualizeResponse(status="ok")
    chart = BarChart(title="t", encoding=PHASE_ENCODING, data=[_phase_row()])
    with pytest.raises(ValidationError):
        VisualizeResponse(status="no_data", visualization=chart)
    assert VisualizeResponse(status="ok", visualization=chart).schema_version == "1.0"
