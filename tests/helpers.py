from app.ctgov.extract import extract
from app.engine.types import FetchedQuery
from app.schemas.plan import Filters
from tests.conftest import load_fixture


def fetched(name: str, label: str | None = None, limit: int | None = None) -> FetchedQuery:
    """A FetchedQuery built from a recorded fixture; `limit` simulates the record cap."""
    fx = load_fixture(name)
    studies = fx["studies"][:limit] if limit else fx["studies"]
    return FetchedQuery(label, Filters(**fx["filters"]), fx["params"], fx["totalCount"], [extract(s) for s in studies])


def trial(nct: int, **protocol) -> dict:
    """A minimal raw study record."""
    ps = {"identificationModule": {"nctId": f"NCT{nct:08d}", "briefTitle": f"Trial {nct}"}}
    ps.update(protocol)
    return {"protocolSection": ps}
