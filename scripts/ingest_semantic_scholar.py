import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.datasets.ingestion import ingest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print("Imported", ingest(args.query, args.limit), "papers")
