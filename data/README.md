# Data storage

Run `python scripts/download_dataset.py --dataset scifact` from the repository root.
The downloader uses the BEIR-hosted archive, records its SHA-256 in `download.json`,
and preserves corpus/query identifiers. `nfcorpus` is also accepted.

Generated dataset folders are ignored by Git. Papers retain only provided metadata;
SciFact's BEIR export does not supply author/year/source URL fields in this application.
Identical text under different IDs is retained to preserve benchmark judgments.
Conflicting duplicate IDs and qrels pointing to removed/unknown documents fail validation.

Semantic Scholar ingestion is separate: it writes `data/semantic-scholar/corpus.jsonl`.
Do not mix its records into a benchmark run. Consult upstream terms before redistribution.
