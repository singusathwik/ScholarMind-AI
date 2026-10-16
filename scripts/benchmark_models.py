import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluation.benchmark import benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON defaults; explicit CLI flags override")
    parser.add_argument("--dataset", choices=["scifact", "nfcorpus"], default="scifact")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--index")
    parser.add_argument("--output")
    parser.add_argument("--modes", nargs="+", choices=["bm25", "dense", "hybrid", "rerank", "optimized"])
    parser.add_argument("--rerank-depth", type=int, default=20)
    parser.add_argument("--constant", type=int, default=60)
    parser.add_argument("--candidates", type=int, default=100)
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        config = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
        allowed = {action.dest for action in parser._actions} - {"config", "help"}
        if set(config) - allowed:
            parser.error("Unknown configuration keys")
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    options = vars(args)
    options.pop("config")
    result = benchmark(**options)
    print(result["summary"])
