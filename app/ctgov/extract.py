"""Normalize raw API study records into `TrialRow`s the engine can group generically.

Every value keeps its provenance: the API field path it came from and the exact raw
value (`excerpt`). Aggregation groups by `key`; citations quote `field` + `excerpt`,
so the citation for a datum is always the literal value that put the trial there.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas.enums import (
    INTERVENTION_TYPE_LABELS,
    PHASE_LABELS,
    SPONSOR_CLASS_LABELS,
    STATUS_LABELS,
    Dimension,
)

P = "protocolSection"
FIELD_PATHS: dict[Dimension, str] = {
    Dimension.PHASE: f"{P}.designModule.phases",
    Dimension.STATUS: f"{P}.statusModule.overallStatus",
    Dimension.START_YEAR: f"{P}.statusModule.startDateStruct.date",
    Dimension.SPONSOR: f"{P}.sponsorCollaboratorsModule.leadSponsor.name",
    Dimension.SPONSOR_CLASS: f"{P}.sponsorCollaboratorsModule.leadSponsor.class",
    Dimension.COUNTRY: f"{P}.contactsLocationsModule.locations[].country",
    Dimension.INTERVENTION: f"{P}.armsInterventionsModule.interventions[].name",
    Dimension.INTERVENTION_TYPE: f"{P}.armsInterventionsModule.interventions[].type",
    Dimension.CONDITION: "derivedSection.conditionBrowseModule.meshes[].term",
}
CONDITION_RAW_PATH = f"{P}.conditionsModule.conditions"
ENROLLMENT_PATH = f"{P}.designModule.enrollmentInfo.count"

MISSING_PHASE_KEY = "MISSING"
MISSING_PHASE_LABEL = "Not specified"
ABSENT_EXCERPT = "(field not present in record)"


@dataclass(frozen=True)
class DimValue:
    key: str  # grouping key: registry enum value, or casefolded name for free text
    label: str  # display label
    field: str  # API field path the value was read from
    excerpt: str  # exact raw value from the API response
    category: str | None = None  # intervention type, used to keep only drugs in networks


@dataclass(frozen=True)
class TrialRow:
    nct_id: str
    title: str | None
    values: dict[Dimension, tuple[DimValue, ...]] = field(default_factory=dict)
    enrollment: int | None = None

    def get(self, dim: Dimension) -> tuple[DimValue, ...]:
        return self.values.get(dim, ())


def _dig(obj: Any, *path: str) -> Any:
    for part in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(part)
    return obj


def _name_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def _display(term: str) -> str:
    """MeSH lowercases some drug names ("pembrolizumab"); capitalize those for display."""
    return term[:1].upper() + term[1:] if term.islower() else term


def _names_match(mesh: str, name: str) -> bool:
    """Whole-word containment either way, so "IV" never matches "Ivermectin"."""

    def contains(haystack: str, needle: str) -> bool:
        return len(needle) >= 4 and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None

    return contains(name, mesh) or contains(mesh, name)


def start_year(date: str | None) -> int | None:
    """Parse `YYYY`, `YYYY-MM` or `YYYY-MM-DD`."""
    match = re.match(r"^(\d{4})", date or "")
    return int(match.group(1)) if match else None


def _phase(ps: dict) -> tuple[DimValue, ...]:
    phases = _dig(ps, "designModule", "phases")
    path = FIELD_PATHS[Dimension.PHASE]
    if not phases:
        # A missing phase is not the same as an explicit "NA" (Not Applicable); keep them apart.
        return (DimValue(MISSING_PHASE_KEY, MISSING_PHASE_LABEL, path, ABSENT_EXCERPT),)
    # Multi-phase trials (e.g. Phase 1/Phase 2) get one combined bucket, so no trial is counted twice.
    key = "/".join(phases)
    label = "/".join(PHASE_LABELS.get(p, p) for p in phases)
    return (DimValue(key, label, path, json.dumps(phases)),)


def _single(value: str | None, dim: Dimension, labels: dict[str, str] | None = None) -> tuple[DimValue, ...]:
    if not value:
        return ()
    label = labels.get(value, value) if labels is not None else value
    key = value if labels is not None else _name_key(value)
    return (DimValue(key, label, FIELD_PATHS[dim], value),)


def _interventions(ps: dict, mesh_terms: list[str]) -> tuple[tuple[DimValue, ...], tuple[DimValue, ...]]:
    """Returns (interventions, intervention_types), each distinct per trial.

    Intervention names are free text ("Pembrolizumab 200 mg IV"). When MeSH terms from
    the record's derived section appear in the name (or vice versa), the MeSH terms become
    the grouping keys, so name variants of the same drug collapse together. A combination
    entry ("Docetaxel and capecitabine") yields one value per drug. The excerpt is always
    the raw name.
    """
    names: dict[str, DimValue] = {}
    types: dict[str, DimValue] = {}
    for item in _dig(ps, "armsInterventionsModule", "interventions") or []:
        raw_name = (item.get("name") or "").strip()
        itype = item.get("type")
        if itype and itype not in types:
            types[itype] = DimValue(
                itype, INTERVENTION_TYPE_LABELS.get(itype, itype), FIELD_PATHS[Dimension.INTERVENTION_TYPE], itype
            )
        if not raw_name:
            continue
        folded = _name_key(raw_name)
        matches = [t for t in mesh_terms if _names_match(t.casefold(), folded)]
        # "Fludarabine phosphate" matches both "Fludarabine" and "fludarabine phosphate": keep the
        # more general term so salt/ester variants group with the parent drug.
        matches = [t for t in matches if not any(o != t and _names_match(o.casefold(), t.casefold()) and len(o) < len(t) for o in matches)]
        resolved = [(t.casefold(), _display(t)) for t in matches] or [(folded, raw_name)]
        for key, label in resolved:
            if key not in names:
                names[key] = DimValue(key, label, FIELD_PATHS[Dimension.INTERVENTION], raw_name, category=itype)
    return tuple(names.values()), tuple(types.values())


def _conditions(study: dict, ps: dict) -> tuple[DimValue, ...]:
    """Prefer normalized MeSH condition terms; fall back to the free-text conditions list."""
    meshes = [m.get("term") for m in _dig(study, "derivedSection", "conditionBrowseModule", "meshes") or [] if m.get("term")]
    if meshes:
        path, values = FIELD_PATHS[Dimension.CONDITION], meshes
    else:
        path, values = CONDITION_RAW_PATH, _dig(ps, "conditionsModule", "conditions") or []
    out: dict[str, DimValue] = {}
    for v in values:
        out.setdefault(_name_key(v), DimValue(_name_key(v), v, path, v))
    return tuple(out.values())


def _countries(ps: dict) -> tuple[DimValue, ...]:
    out: dict[str, DimValue] = {}
    for loc in _dig(ps, "contactsLocationsModule", "locations") or []:
        country = loc.get("country")
        if country and country not in out:
            out[country] = DimValue(country, country, FIELD_PATHS[Dimension.COUNTRY], country)
    return tuple(out.values())


def extract(study: dict[str, Any]) -> TrialRow:
    ps = study.get("protocolSection", {})
    nct_id = _dig(ps, "identificationModule", "nctId")
    start_date = _dig(ps, "statusModule", "startDateStruct", "date")
    year = start_year(start_date)
    sponsor = _dig(ps, "sponsorCollaboratorsModule", "leadSponsor") or {}
    intervention_mesh = [
        m.get("term") for m in _dig(study, "derivedSection", "interventionBrowseModule", "meshes") or [] if m.get("term")
    ]
    interventions, intervention_types = _interventions(ps, intervention_mesh)
    enrollment = _dig(ps, "designModule", "enrollmentInfo", "count")

    values = {
        Dimension.PHASE: _phase(ps),
        Dimension.STATUS: _single(_dig(ps, "statusModule", "overallStatus"), Dimension.STATUS, STATUS_LABELS),
        Dimension.START_YEAR: (
            (DimValue(str(year), str(year), FIELD_PATHS[Dimension.START_YEAR], start_date),) if year else ()
        ),
        Dimension.SPONSOR: _single(sponsor.get("name"), Dimension.SPONSOR),
        Dimension.SPONSOR_CLASS: _single(sponsor.get("class"), Dimension.SPONSOR_CLASS, SPONSOR_CLASS_LABELS),
        Dimension.COUNTRY: _countries(ps),
        Dimension.INTERVENTION: interventions,
        Dimension.INTERVENTION_TYPE: intervention_types,
        Dimension.CONDITION: _conditions(study, ps),
    }
    return TrialRow(
        nct_id=nct_id,
        title=_dig(ps, "identificationModule", "briefTitle"),
        values=values,
        enrollment=enrollment if isinstance(enrollment, int) else None,
    )
