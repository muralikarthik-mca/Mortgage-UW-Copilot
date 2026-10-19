# Pinecone migration verification

The dedicated `mortgage-uw-copilot` cosine index stores 99 vectors with 768 dimensions, generated locally by Ollama `nomic-embed-text`. Existing embeddings were reused during migration.

| Corpus | Chunking | Vectors |
|---|---|---:|
| Fictional training | Fixed | 7 |
| Fannie Mae | Fixed | 31 |
| Fictional training | Semantic | 13 |
| Fannie Mae | Semantic | 48 |

Each corpus/strategy uses a versioned namespace bound to source metadata, chunk fingerprint, chunking version, and embedding model. No loan JSON is uploaded. Full original chunk provenance is stored with the vectors and retained in the local source snapshot. Failed writes do not replace the active pointer. Obsolete namespaces are retained for explicit cleanup.

`pinecone_verification.json` records 15 actual local-versus-Pinecone comparisons: five representative questions across dense, hybrid, and hybrid-with-reranking. All returned identical ordered top-five chunk IDs. The adapter makes actual Pinecone dense queries; BM25 and passage reranking remain local. BM25-only candidates have null dense scores/ranks because Pinecone did not return them.

The complete 30-question, six-configuration benchmark and eight representative answer checks were rerun with Pinecone. All six configurations retained their local baseline Precision@5 and Recall@5. Semantic + dense measured Precision@5 0.2067, Recall@5 1.0000, median retrieval latency 178.4 ms, and P95 349.4 ms. Five answerable samples scored 1.0 on automated support precision and required-section coverage; three ambiguous/absent-topic samples correctly abstained. Actual source-fidelity/citation scores and answer outputs are in `rag_comparison.json`.

Pre-migration local measurements remain in `pre_pinecone_rag_comparison.*`. Latencies are independent single runs, affected by hardware, network, and concurrent load; they are not a controlled speed comparison. Retrieval latency excludes chat generation. The eight answer checks are a development subset, not held-out validation or underwriting applicability judgments.

The SDK installed successfully in the existing `.venv314` Python 3.14 environment. `pip check` reported no broken requirements. Unit and adapter results are in `test_results.*`; browser results are in `browser_test_results.*` and explicitly identify the configured backend.

The local `.env` selects Pinecone and is ignored by Git. The tracked template contains no API key. Changing the key/backend requires restarting Streamlit. Use `VECTOR_BACKEND=local` for an explicit rollback. Pinecone RAG requires network access and a valid API key; no automatic fallback conceals database failures. Local snapshots are still required for BM25, source validation, and citation rendering. This is an incremental local-demo migration, not a distributed ingestion or authentication system.
