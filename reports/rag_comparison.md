# Measured RAG comparison

Measured at 2026-10-19T02:54:38.681125+00:00 UTC; 30 labeled questions.
Models: embeddings `nomic-embed-text`; chat `gemma3:1b`.

Vector backend: `pinecone`. BM25 and passage reranking run in Python.

| Chunking | Retrieval | Precision@5 | Recall@5 | Median ms | P95 ms |
|---|---|---:|---:|---:|---:|
| fixed | dense | 0.213 | 0.908 | 184.4 | 280.7 |
| fixed | hybrid | 0.240 | 0.965 | 188.8 | 284.6 |
| fixed | hybrid_rerank | 0.233 | 0.963 | 190.1 | 289.2 |
| semantic | dense | 0.207 | 1.000 | 178.4 | 349.4 |
| semantic | hybrid | 0.200 | 0.986 | 190.5 | 225.4 |
| semantic | hybrid_rerank | 0.173 | 0.875 | 212.3 | 308.6 |

Chunk-level relevance uses section IDs and prewritten source-text anchors. Precision uses five slots; missing slots count as nonrelevant. Recall excludes no-answer cases (undefined, never scored as 1). Overlap can create multiple gold chunks, and changing chunk granularity changes the recall denominator.

Latency is one wall-clock sample per question/configuration, including index validation/loading, Ollama query embedding, and ranking. It excludes index building and chat generation. No parallel requests or query-vector cache were used. Hardware/load and first-request warmup are not controlled.

Dense = cosine search. Hybrid = independent dense/BM25 candidate union plus reciprocal-rank fusion. Hybrid rerank = the hybrid pool followed by a deterministic query-passage lexical scorer, not a cross-encoder. The JSON contains raw scores and all before/after candidate ranks.

## Answer evaluation

8 actual pipeline answers measured on semantic + dense (the deployed default) only; 5 invoked Ollama and the remainder used documented corpus-scope abstention. Answer scores are not a six-configuration answer comparison.

| Case | Source fidelity | Citation correctness | Support precision | Required section coverage | Correct abstention |
|---|---:|---:|---:|---:|---|
| A01 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| A05 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| A08 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| A12 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| T04 | 1.00 | 1.00 | 1.00 | 1.00 | n/a |
| U01 | 1.00 | 1.00 | 1.00 | n/a | True |
| N01 | 1.00 | 1.00 | 1.00 | n/a | True |
| N03 | 1.00 | 1.00 | 1.00 | n/a | True |

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

## Limitations

- Hand-labeled, small corpus; no independent held-out set. Relevance anchors are not underwriting certifications.
- Missing multiple-job, self-employment calculation, FHA, and other topics cannot be answered. Raw nearest-neighbor retrieval still returns candidates; answer-stage abstention is tested separately.
- Verified snapshots do not establish effective-date applicability; see `source_quality.json`.
- Semantic segmentation is deterministic TextTiling-inspired lexical cohesion, not embedding segmentation. Page/table boundaries remain limitations. Same-page exceptions attach to preceding text.
- Reranking is lexical, not a learned cross-encoder. The small chat model can select irrelevant evidence or fail to abstain; exact-source fidelity can be perfect while answer relevance is poor.
- Latency is single-run and load-dependent. Cached reports validate source/code/dataset hashes and model names, not silent same-name model replacement. Measured model digests are recorded.
- The initial measured diagnostics are retained in `initial_rag_comparison.json` and `.md`. Subsequent source-option numbering, top-five answer context, and conservative missing-topic/ambiguity guards address observed failures; this is development-set iteration, not held-out validation. Guards describe corpus coverage, not underwriting rules. Unrecognized out-of-scope requests can still escape them.
