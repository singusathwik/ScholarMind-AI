# Dataset card

## SciFact / BEIR retrieval export

Origin: [SciFact](https://github.com/allenai/scifact), repackaged by
[BEIR](https://github.com/beir-cellar/beir). Retrieved 2026-10-15 from the BEIR-hosted archive.
Archive SHA-256: `536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165`.
Normalized corpus fingerprint: `9e76f432981aaa1c451143ef178d40c81e5c5c155588b061a9cc74d03eeddf30`.

Observed counts: 5,183 documents, 809 train queries with judgments, 300 test queries with
judgments; train/test query-ID intersection is empty. No empty documents or duplicate IDs
were removed. Mean combined title/abstract length is 214.63 whitespace tokens; maximum 1,541.
Token counts depend on tokenization: MiniLM's mean is 337.20 model tokens including specials.
3,681 combined texts exceed its configured 256-token limit.

Internal fields are paper_id, title, abstract, source, authors, publication_year, URL and
fields_of_study. Only the first four are populated from this export. The corpus contains
abstracts, not full papers. Scientific claims are the retrieval queries; they are not all
natural search questions or true statements. Relevance identifies evidence-bearing documents,
not endorsement of the claim. The app is a literature finder, not a fact verifier.

NFKC and whitespace normalization preserve IDs. Same-ID exact duplicates are collapsed;
conflicting duplicates fail. Same text with different IDs stays. Empty records are excluded
and any resulting dangling qrels cause failure instead of silent benchmark changes.

## License and use

The [original SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md)
distinguishes claims/evidence annotations (CC BY 4.0), abstracts from S2ORC (ODC-By 1.0),
and repository code (Apache 2.0). The [BEIR Hugging Face card](https://huggingface.co/datasets/BeIR/scifact)
separately labels its package CC BY-SA 4.0. These upstream notices are not treated as
interchangeable. Dataset contents are downloaded locally and excluded from this project's
MIT-licensed source repository; check the applicable upstream conditions before redistribution.

## Limits

Small, largely biomedical, English corpus; sparse relevance judgments; incomplete author/year
metadata; possible overlap with pretrained web/scientific text. No claim of benchmark-wide
training decontamination. Results do not generalize automatically to other fields or languages.

NFCorpus has a shared download/load path. Its download and validation produced 3,633 documents
and 323 test queries; its full neural benchmark is optional and has not been executed.
Semantic Scholar is a separate discovery corpus with API-specific terms.
The live unauthenticated ingestion attempt on 2026-10-15 returned HTTP 429 after bounded
retries; fixture tests verify parsing/caching without claiming a successful live import.
