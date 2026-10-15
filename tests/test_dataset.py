import json

import pytest

from ml.datasets.loader import load_corpus, load_judgments, normalize


def test_normalize():
    assert normalize(" Ａ\n  cell ") == "A cell"


def test_beir_metadata_url(tmp_path):
    row = {
        "_id": "MED-1",
        "title": "Title",
        "text": "Abstract",
        "metadata": {"url": "https://example.org/paper"},
    }
    (tmp_path / "corpus.jsonl").write_text(json.dumps(row))
    papers, _ = load_corpus(tmp_path)
    assert papers[0].url == "https://example.org/paper"


def test_duplicate_and_empty(tmp_path):
    row = {"_id": "a", "title": "A cell", "text": "biology"}
    (tmp_path / "corpus.jsonl").write_text("\n".join(map(json.dumps, [row, row, {"_id": "b"}])))
    papers, stats = load_corpus(tmp_path)
    assert len(papers) == 1
    assert stats["duplicates_handled"] == stats["empty_removed"] == 1
    assert papers[0].publication_year is None


def test_conflicting_ids(tmp_path):
    (tmp_path / "corpus.jsonl").write_text('{"_id":"a","title":"one"}\n{"_id":"a","title":"two"}')
    with pytest.raises(ValueError, match="Conflicting"):
        load_corpus(tmp_path)


def test_qrels_integrity(tmp_path, papers):
    (tmp_path / "queries.jsonl").write_text('{"_id":"q","text":"question"}')
    (tmp_path / "qrels").mkdir()
    path = tmp_path / "qrels/test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq\ta\t1\n")
    queries, qrels = load_judgments(tmp_path, papers)
    assert queries == {"q": "question"} and qrels == {"q": {"a": 1}}
    path.write_text("query-id\tcorpus-id\tscore\nq\tmissing\t1\n")
    with pytest.raises(ValueError, match="Dangling"):
        load_judgments(tmp_path, papers)
