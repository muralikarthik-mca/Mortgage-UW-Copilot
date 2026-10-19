# Mortgage Underwriting Copilot

Local Streamlit starter using five synthetic loans, supplied Fannie Mae PDF snapshots, and separately selected fictional training policies. This is a review demo, not an eligibility or credit decision system. No real borrower data is included.

## Run on Windows

Use the existing Python 3.14 environment explicitly (also configured in `.vscode/settings.json`):

```powershell
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Ollama must be running locally with `gemma3:1b` and `nomic-embed-text` pulled. `.env` is optional; defaults match `.env.example`. Configure `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBED_MODEL` there to override defaults. Cloud APIs are not used.

The default URL is `http://127.0.0.1:11434`, which avoids the observed Windows IPv6 fallback delay with `localhost`. Explicit environment overrides are still respected.

Use the persistent left navigation to open **Dashboard**, **Loans**, **Guidelines**, or **RAG Evaluation**. Select the active loan and guideline source in the sidebar. Open **Guidelines** and click **Build / rebuild guideline index** for each source/strategy you want to use.

The main workspace shows loan summary cards, income details, document inventory, and preliminary findings. Open the **AI assistant** button at the bottom right to ask **Review the income documentation and identify outstanding items.** The 400px desktop panel floats above the page; on narrow screens it fills almost the whole viewport. Close it with the close control, launcher, or Escape. Keyboard focus enters the question field on open and returns to the launcher on close; mobile keyboard focus stays inside the panel.

Conversation history and unsent drafts survive minimizing. Histories are separated by loan, source, chunking strategy, and reranking setting, and remain available when switching back during the same Streamlit session. A browser refresh starts a new Streamlit session. Each request includes the server-selected loan, current view, retrieval settings, income context, and recent same-context conversation. Frontend events cannot select a different backend loan; explicit cross-loan requests and stale-context events are rejected. Only the active context's history is sent to the component. These controls enforce selected-loan isolation within this synthetic demo; they do not add user authentication or production access control.

Reviews run in background workers, so minimizing does not cancel them. Loading, request error, and insufficient-evidence notices appear inside the panel. Answers include clickable agency citations, expandable full sources, and an evidence audit trail. The chat assets are isolated under `ui/assistant.*`; `assistant_ui.py` uses Streamlit 1.66's bidirectional component API, and `chat_service.py` validates requests before calling the existing `build_graph().invoke(...)` backend. No separate frontend server or disconnected chat mockup is used.

## Data and ingestion

`data/loans/loans.json` contains LN-10045 (salaried), LN-10046 (salaried with an older paystub), LN-10047 (bonus), LN-10048 (multiple jobs), and LN-10049 (self-employed). Every amount remains UNVERIFIED. Document contents are absent; RECEIVED does not establish adequacy.

`data/guidelines/TRAINING_ONLY_POLICIES.md` contains seven explicitly fictional TR-INC sections. Its 60-day recency flag is confined to training mode and is never used as an agency rule.

`data/guidelines/agency/fannie_mae/` retains the four original clipped browser-print PDFs unchanged. Its `verified/` directory contains archived official HTML, isolated complete topic bodies, and width-safe replacement PDFs. `manifest.json` binds originals, raw HTML, article HTML, and active PDFs by SHA-256. All four active PDFs passed 100% body-token coverage against their official topic bodies, including tables, notes, alternatives, and exceptions. Token coverage does not establish table reading order or completeness of the entire Selling Guide. Topic revision dates are stored separately from `effective_date: null`; no effective date is invented. `reports/source_quality.json` documents clipping, readable original revisions, verified revisions, missing-topic coverage, and source verification times.

`core.py` extracts each active PDF page under its topic identifier. `chunking.py` offers two genuinely different boundary algorithms:

- **Fixed baseline:** 800-character windows, 150-character overlap. It can split sentences or conditions; it preserves all source characters through overlap.
- **Semantic:** deterministic [TextTiling-inspired](https://aclanthology.org/J97-1003/) segmentation based on TF-IDF lexical cohesion between adjacent two-sentence windows, paragraph boundaries, and topic-cohesion drops. It uses meaning-related topic vocabulary rather than larger character windows. Notes, exceptions, alternative paths, and lists following a colon stay with preceding text on the same page. It does not call Ollama during segmentation or claim to be an embedding-based segmenter. Page-spanning conditions and complex tables still require adjacent-page review.

Every chunk preserves original text, section ID, page reference, character offsets, URL, revision, unknown effective date, source hashes, quality warnings, boundary reason/cohesion, and adjacent chunk IDs. Semantic text is not paraphrased or deduplicated. Tests verify topic-driven boundary changes, lossless coverage, exception attachment, and distinct real-corpus boundaries. Manifest mismatches fail ingestion. Refreshing official sources is an explicit maintenance step, never a startup side effect:

```powershell
.\.venv314\Scripts\python.exe refresh_sources.py
.\.venv314\Scripts\python.exe audit_sources.py
```

Refreshing requires optional browser dependencies and installed Edge. It retains originals, verifies the official topic identity, archives provenance, and refuses to update the manifest if printed body-token coverage is below 100%. Existing readable PDF revisions matched the checked online topics; the clipped employment-documentation heading cannot establish its original revision independently. Verification is a dated snapshot, not a guarantee of future currency.

Ollama `nomic-embed-text` creates all embeddings with document/query task prefixes. `VECTOR_BACKEND=local` uses the original NumPy search; `VECTOR_BACKEND=pinecone` uses hosted Pinecone dense search. Separate JSON snapshots under `.indexes/` retain original chunks, metadata, vectors for rollback, and active remote namespace pointers. Schema/chunking/source-signature validation rejects outdated indexes. The retrieval dropdown offers:

1. **Dense vector:** cosine ranking from a dense candidate pool.
2. **Hybrid dense + BM25:** union of independently retrieved dense and BM25 candidates, fused using reciprocal ranks. BM25 can introduce candidates outside the dense pool.
3. **Hybrid + passage reranking:** the hybrid candidate pool reordered by an explicit query-passage lexical scorer using best-sentence term coverage, query bigrams, and whole-passage coverage. This is not a learned cross-encoder or an underwriting confidence score.

All candidate IDs, raw dense/BM25 scores, independent ranks, RRF scores, passage scores, and before/after ranks are available in the assistant audit and evaluation dashboard. Indexes rebuild atomically and reject changed source content, metadata, chunking version, or embedding model name. The `build_graph().invoke(...)` interface is preserved; it remains a synchronous Python workflow, not a LangGraph dependency. Simple paystub-age questions use direct loan-date calculation and a short cited answer, without RAG or Ollama.

The UI defaults to semantic chunking with dense retrieval because it had the highest measured Recall@5 on this curated set. Hybrid and passage reranking remain explicit options; the measured report retains their regressions rather than assuming that more ranking stages are better. This is a demo-specific choice, not evidence of general superiority.

The small chat model selects short numbered evidence options through structured output; Python resolves them to validated original chunk IDs. Retrieval returns five passages for the source audit. Answer selection conservatively uses the highest-ranked passage, or the first two for explicitly comparative questions mentioning sections, topics, both, or comparison. This limits unrelated lower-ranked selections; the model can still abstain. It can miss answers requiring more passages or implicit comparisons. The app quotes original source text with section citations and renders loan observations directly from JSON with `[loan:LOAN_ID]` citations. Conservative corpus-scope guards abstain for recognized missing topics or vague first-turn questions; they describe available documents, not underwriting rules. No supporting selection produces an insufficient-evidence response. Human review remains required.

## Verification

```powershell
.\.venv314\Scripts\python.exe -m pytest -q
.\.venv314\Scripts\python.exe -m pip check
.\.venv314\Scripts\python.exe evaluate.py
.\.venv314\Scripts\python.exe smoke_workflow.py
```

For browser verification, install the optional development dependency and run against the app on port 8502 (the check uses installed Microsoft Edge):

```powershell
.\.venv314\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv314\Scripts\python.exe verify_ui.py
```

The 31 baseline tests passed before changes; the checkpoint tag is `checkpoint/pre-rag-improvements-20261018`. Existing tests are retained, with additional tests for semantic boundaries, exception preservation, hybrid candidate expansion, reranking traces, complete-source provenance, and metric/rubric behavior. `smoke_workflow.py` checks rendering, index building, and the live loan-scoped backend. `verify_ui.py` checks the actual desktop/mobile browser UI, context isolation, source links, insufficient evidence, history, focus, and evaluation navigation. Runtime screenshots remain under ignored `artifacts/`.

## Measured RAG evaluation

`data/evaluation/questions.json` contains 30 labeled questions: salary, employment documentation, bonus/overtime, multiple income sources, exceptions, multi-section questions, ambiguity, and absent topics. Labels are source-text anchors scoped to section IDs. They are not invented agency requirements.

```powershell
# Return a matching cached measurement or perform the six-way retrieval benchmark:
.\.venv314\Scripts\python.exe evaluate.py
# Force fresh retrieval measurements and eight representative live answer checks:
.\.venv314\Scripts\python.exe evaluate.py --force --answers
```

The comparison covers two chunking algorithms × three retrieval modes. It records chunk-level Precision@5, Recall@5, and measured end-to-end retrieval latency. Recall is undefined for unanswerable questions and excluded from its macro average. Answer checks measure exact-source faithfulness, citation correctness, independent evidence relevance, multi-section coverage, and abstention using `reports/evaluation_rubric.md`. Source fidelity is deliberately narrower than human semantic entailment; an irrelevant exact quote can score perfectly on fidelity while failing relevance. Eight pipeline answer cases use semantic + dense, matching the deployed default; the report distinguishes actual Ollama calls from scope-guard abstention. The original measured failures are retained in `reports/initial_rag_comparison.*`. Scope guards and a narrower ranked answer window address observed development-set failures; these are not held-out validation results.

The pre-Pinecone verification passed **48 tests** and desktop/mobile browser checks. Its local semantic + dense retrieval measured Precision@5 **0.207**, Recall@5 **1.000**, and median retrieval latency **125.3 ms** across 30 questions. Five answerable samples achieved support precision and required-section coverage of 1.0; three ambiguous/absent-topic samples correctly abstained. These eight answer checks do not certify unseen questions. `reports/answer_relevance_probe.json` records the five-case check performed before that benchmark. Current database measurements are identified separately below and in the comparison report.

Read `reports/rag_comparison.md` for measured results and failure analysis; its companion JSON contains raw scores, candidate traces, and actual answer outputs. The evaluation dashboard loads matching measured results quickly and exposes the report, answer rubric scores, and before/after candidate ranks. Fresh benchmarks require Ollama and can take several minutes. The local JSON index is sufficient for this small corpus; no infrastructure migration is justified.

## Pinecone setup

Use the existing `.venv314`; `requirements.txt` includes the Pinecone SDK. Copy `.env.example` to `.env` only if `.env` is absent. Put the API key in `.env` only; never in the tracked template or source files. Configure:

```dotenv
PINECONE_API_KEY=your_private_key
VECTOR_BACKEND=pinecone
PINECONE_INDEX=mortgage-uw-copilot
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE_PREFIX=mortgage-uw
```

```powershell
.\.venv314\Scripts\python.exe -X utf8 pinecone_setup.py --migrate
.\.venv314\Scripts\python.exe -X utf8 pinecone_setup.py --verify --activate
```

Migration reuses valid local embeddings, creates a dedicated cosine index matching their actual dimensions (768 in the measured migration), and uploads separate versioned namespaces for each corpus/strategy. Source text and lossless provenance accompany the vectors. It verifies fetch/search visibility before publishing the local namespace pointer. Failed writes preserve the previous pointer; previous namespaces are retained, not deleted. New source/model versions get new namespaces. Inspect or clean up superseded namespaces separately as the corpus grows.

Restart Streamlit after changing `.env`. Guidelines shows the active vector backend. Dense candidates come from Pinecone queries; BM25, reciprocal-rank fusion, and passage reranking remain local. BM25-only candidates have no returned dense score/rank, represented as null rather than an invented score. Citation text is resolved from the matching local source snapshot, so local source files/snapshots remain required. Loan JSON is not uploaded. Pinecone receives guideline text/vectors and query embeddings; network access is required for RAG, while Ollama remains local. No automatic fallback hides Pinecone failures.

Set `VECTOR_BACKEND=local` and restart to use the retained local indexes. Ordinary pytest tests force the local backend or mock the adapter; `pinecone_setup.py --verify` performs live database comparisons. `reports/pinecone_migration.json` records uploaded counts and namespaces; `reports/pinecone_verification.json` records the 15 measured local/Pinecone comparisons. `reports/pre_pinecone_rag_comparison.*` preserves pre-migration measurements, while the current comparison report explicitly identifies its backend. Run `evaluate.py --force --answers` for measurements in your environment. Keep `.env`, database caches, and credentials out of Git; `.env.example` contains blank credentials only.

Pinecone verification: **58 tests passed**, with the desktop/mobile browser workflow passing against the active Pinecone app. All 15 local/Pinecone retrieval comparisons returned identical ordered top-five IDs. The complete benchmark retained its retrieval quality scores; semantic + dense measured median retrieval latency **178.4 ms**. All eight representative answer checks passed their automated rubric expectations. See `reports/pinecone_migration.md` and the raw reports for details and limits.

## Five-minute demo

1. Start Ollama and the app with `.venv314`. Build the selected index in Guidelines if needed; a benchmark builds all four indexes.
2. Select LN-10046 and ask **How old is the paystub?** Confirm the short, loan-cited 151-day calculation.
3. Ask **What does the documentation section say about the date of the most recent paystub relative to initial application?** Open the official section citation and evidence/ranking audit.
4. Try requesting another loan ID; the assistant rejects it. Switch to LN-10049 and ask paystub age to show an explicit missing-evidence response.
5. Open RAG Evaluation. Compare the six measured configurations, candidate rank changes, and the documented answer failures. Cached measurements are labeled with their actual timestamp.

## Remaining limitations

The corpus covers only four official topics and seven fictional sections. Missing multiple-job, self-employment-calculation, FHA, and referenced topics require more verified sources. Effective-date applicability is unknown. Lexical semantic segmentation is not an embedding model; page boundaries and tables can separate conditions. The passage reranker is not a cross-encoder. The small chat model can select irrelevant evidence or fail to abstain. Gold labels are curated, overlap affects chunk-level recall denominators, and there is no held-out evaluation or human policy-applicability scoring. Latencies are single-run and hardware/load-dependent. Model names are validated, but silent replacement under the same name requires a manual rebuild; measured model digests are recorded. The demo has selected-loan isolation but no production authentication or credit-decision capability.

The imported starter had misplaced contents in `requirements.txt`, `core.py`, `evaluate.py`, and `README.md`; these files were repaired. The original loan JSON, fictional policy file, and PDFs were retained. The root `test_core.py` is a misplaced duplicate loan list; pytest uses the actual tests under `tests/`. Importlib mode avoids the duplicate test module name collision.
