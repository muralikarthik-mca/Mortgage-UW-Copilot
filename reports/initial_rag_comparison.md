# Measured RAG comparison

Measured at 2026-10-18T22:03:41.786285+00:00 UTC; 30 labeled questions.
Models: embeddings `nomic-embed-text`; chat `gemma3:1b`.

| Chunking | Retrieval | Precision@5 | Recall@5 | Median ms | P95 ms |
|---|---|---:|---:|---:|---:|
| fixed | dense | 0.213 | 0.908 | 125.3 | 176.8 |
| fixed | hybrid | 0.240 | 0.965 | 128.2 | 177.9 |
| fixed | hybrid_rerank | 0.233 | 0.963 | 133.3 | 165.7 |
| semantic | dense | 0.207 | 1.000 | 131.7 | 171.0 |
| semantic | hybrid | 0.200 | 0.986 | 133.2 | 155.1 |
| semantic | hybrid_rerank | 0.173 | 0.875 | 136.2 | 181.6 |

Chunk-level relevance uses section IDs and prewritten source-text anchors. Precision uses five slots; missing slots count as nonrelevant. Recall excludes no-answer cases (undefined, never scored as 1). Overlap can create multiple gold chunks, and changing chunk granularity changes the recall denominator.

Latency is one wall-clock sample per question/configuration, including index validation/loading, Ollama query embedding, and ranking. It excludes index building and chat generation. No parallel requests or query-vector cache were used. Hardware/load and first-request warmup are not controlled.

Dense = cosine search. Hybrid = independent dense/BM25 candidate union plus reciprocal-rank fusion. Hybrid rerank = the hybrid pool followed by a deterministic query-passage lexical scorer, not a cross-encoder. The JSON contains raw scores and all before/after candidate ranks.

## Answer evaluation

8 actual answers measured on semantic + hybrid_rerank only. Answer scores are not a six-configuration answer comparison.

| Case | Source fidelity | Citation correctness | Support precision | Required section coverage | Correct abstention |
|---|---:|---:|---:|---:|---|
| A01 | 1.00 | 1.00 | 0.50 | 1.00 | n/a |
| A05 | 1.00 | 1.00 | 0.50 | 1.00 | n/a |
| A08 | 1.00 | 1.00 | 0.50 | 0.50 | n/a |
| A12 | 1.00 | 1.00 | 0.00 | 0.00 | n/a |
| T04 | 1.00 | 1.00 | 0.50 | 1.00 | n/a |
| U01 | 1.00 | 1.00 | 0.00 | n/a | False |
| N01 | 1.00 | 1.00 | 0.00 | n/a | False |
| N03 | 1.00 | 1.00 | 0.00 | n/a | False |

Rubric: exact quote fidelity and authoritative record equality; citation ID/section/page/version and display checks; independent gold-anchor relevance; required-section coverage; and no-answer abstention. These are automated source-fidelity checks, not human semantic entailment or policy applicability. See `evaluation_rubric.md`.

## Failure analysis

- fixed/dense A01: recall 0.500; found 1 of 2 gold chunks. Top sections: B3-3.3-01, B3-3.2-01, B3-3.1-01, B3-3.2-01, B3-3.1-01.
- fixed/dense A08: recall 0.800; found 4 of 5 gold chunks. Top sections: B3-3.1-01, B3-3.2-01, B3-3.2-01, B3-3.2-01, B3-3.1-01.
- fixed/dense A14: recall 0.333; found 1 of 3 gold chunks. Top sections: B3-3.1-01, B3-3.1-01, B3-3.1-01, B3-3.3-02, B3-3.3-02.
- fixed/dense A16: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.2-01, B3-3.2-01, B3-3.3-01, B3-3.3-02, B3-3.3-02.
- fixed/dense A17: recall 0.500; found 2 of 4 gold chunks. Top sections: B3-3.1-01, B3-3.1-01, B3-3.3-02, B3-3.3-02, B3-3.1-01.
- fixed/hybrid A14: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.1-01, B3-3.1-01, B3-3.1-01, B3-3.3-01, B3-3.3-02.
- fixed/hybrid A17: recall 0.500; found 2 of 4 gold chunks. Top sections: B3-3.1-01, B3-3.1-01, B3-3.3-02, B3-3.3-01, B3-3.1-01.
- fixed/hybrid_rerank A08: recall 0.600; found 3 of 5 gold chunks. Top sections: B3-3.1-01, B3-3.2-01, B3-3.2-01, B3-3.1-01, B3-3.2-01.
- fixed/hybrid_rerank A17: recall 0.500; found 2 of 4 gold chunks. Top sections: B3-3.1-01, B3-3.3-02, B3-3.3-01, B3-3.3-01, B3-3.1-01.
- semantic/hybrid A17: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.1-01, B3-3.3-02, B3-3.1-01, B3-3.3-01, B3-3.3-01.
- semantic/hybrid_rerank A07: recall 0.000; found 0 of 1 gold chunks. Top sections: B3-3.2-01, B3-3.2-01, B3-3.3-01, B3-3.3-01, B3-3.3-02.
- semantic/hybrid_rerank A08: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.1-01, B3-3.2-01, B3-3.1-01, B3-3.2-01, B3-3.1-01.
- semantic/hybrid_rerank A10: recall 0.000; found 0 of 1 gold chunks. Top sections: B3-3.3-02, B3-3.3-02, B3-3.1-01, B3-3.2-01, B3-3.2-01.
- semantic/hybrid_rerank A16: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.3-01, B3-3.3-01, B3-3.2-01, B3-3.3-02, B3-3.2-01.
- semantic/hybrid_rerank A17: recall 0.667; found 2 of 3 gold chunks. Top sections: B3-3.1-01, B3-3.3-02, B3-3.3-01, B3-3.3-01, B3-3.1-01.
- Answer A01: support precision 0.500, section coverage 1.0, correct abstention None. Inspect the actual answer/selection in the JSON report.
- Answer A05: support precision 0.500, section coverage 1.0, correct abstention None. Inspect the actual answer/selection in the JSON report.
- Answer A08: support precision 0.500, section coverage 0.5, correct abstention None. Inspect the actual answer/selection in the JSON report.
- Answer A12: support precision 0.000, section coverage 0.0, correct abstention None. Inspect the actual answer/selection in the JSON report.
- Answer T04: support precision 0.500, section coverage 1.0, correct abstention None. Inspect the actual answer/selection in the JSON report.
- Answer U01: support precision 0.000, section coverage None, correct abstention False. Inspect the actual answer/selection in the JSON report.
- Answer N01: support precision 0.000, section coverage None, correct abstention False. Inspect the actual answer/selection in the JSON report.
- Answer N03: support precision 0.000, section coverage None, correct abstention False. Inspect the actual answer/selection in the JSON report.

## Limitations

- Hand-labeled, small corpus; no independent held-out set. Relevance anchors are not underwriting certifications.
- Missing multiple-job, self-employment calculation, FHA, and other topics cannot be answered. Raw nearest-neighbor retrieval still returns candidates; answer-stage abstention is tested separately.
- Verified snapshots do not establish effective-date applicability; see `source_quality.json`.
- Semantic segmentation is deterministic TextTiling-inspired lexical cohesion, not embedding segmentation. Page/table boundaries remain limitations. Same-page exceptions attach to preceding text.
- Reranking is lexical, not a learned cross-encoder. The small chat model can select irrelevant evidence or fail to abstain; exact-source fidelity can be perfect while answer relevance is poor.
- Latency is single-run and load-dependent. Cached reports validate source/code/dataset hashes and model names, not silent same-name model replacement. Measured model digests are recorded.
