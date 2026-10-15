import httpx

from ml.datasets.ingestion import ingest
from ml.datasets.loader import load_corpus


def test_optional_ingestion_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "paperId": "external",
                        "title": "Real API-shaped fixture",
                        "abstract": None,
                        "authors": [],
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert ingest("test", output=tmp_path / "corpus", client=client) == 1
    assert ingest("test", output=tmp_path / "corpus") == 1
    assert len(calls) == 1
    papers, _ = load_corpus(tmp_path / "corpus")
    assert papers[0].publication_year is None and papers[0].abstract == ""
