import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.pipelines.build_index import build
from ml.retrieval.dense import MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--representation", choices=["title", "abstract", "both"], default="both")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output")
    parser.add_argument("--revision")
    parser.add_argument("--reranker-revision")
    args = parser.parse_args()
    print(build(**vars(args)))
