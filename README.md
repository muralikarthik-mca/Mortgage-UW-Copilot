# Mortgage Underwriting Copilot

Local Streamlit starter using five synthetic loans, supplied Fannie Mae PDF snapshots, and separately selected fictional training policies. This is a review demo, not an eligibility or credit decision system. No real borrower data is included.

## Run on Windows

Use the existing Python 3.14 environment explicitly (also configured in `.vscode/settings.json`):

```powershell
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Ollama must be running locally with `gemma3:1b` and `nomic-embed-text` pulled. `.env` is optional; defaults match `.env.example`. Configure `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, and `OLLAMA_EMBED_MODEL` there to override defaults. Cloud APIs are not used.

Use the persistent left navigation to open **Dashboard**, **Loans**, **Guidelines**, or **RAG Evaluation**. Select the active loan and guideline source in the sidebar. Open **Guidelines** and click **Build / rebuild guideline index** for each source/strategy you want to use.

The main workspace shows loan summary cards, income details, document inventory, and preliminary findings. Open the **AI assistant** button at the bottom right to ask **Review the income documentation and identify outstanding items.** The 400px desktop panel floats above the page; on narrow screens it fills almost the whole viewport. Close it with the close control, launcher, or Escape. Keyboard focus enters the question field on open and returns to the launcher on close; mobile keyboard focus stays inside the panel.

Conversation history and unsent drafts survive minimizing. Histories are separated by loan, source, chunking strategy, and reranking setting, and remain available when switching back during the same Streamlit session. A browser refresh starts a new Streamlit session. Each request includes the server-selected loan, current view, retrieval settings, income context, and recent same-context conversation. Frontend events cannot select a different backend loan; explicit cross-loan requests and stale-context events are rejected. Only the active context's history is sent to the component. These controls enforce selected-loan isolation within this synthetic demo; they do not add user authentication or production access control.

Reviews run in background workers, so minimizing does not cancel them. Loading, request error, and insufficient-evidence notices appear inside the panel. Answers include clickable agency citations, expandable full sources, and an evidence audit trail. The chat assets are isolated under `ui/assistant.*`; `assistant_ui.py` uses Streamlit 1.66's bidirectional component API, and `chat_service.py` validates requests before calling the existing `build_graph().invoke(...)` backend. No separate frontend server or disconnected chat mockup is used.

## Data and ingestion

`data/loans/loans.json` contains LN-10045 (salaried), LN-10046 (salaried with an older paystub), LN-10047 (bonus), LN-10048 (multiple jobs), and LN-10049 (self-employed). Every amount remains UNVERIFIED. Document contents are absent; RECEIVED does not establish adequacy.

`data/guidelines/TRAINING_ONLY_POLICIES.md` contains seven explicitly fictional TR-INC sections. Its 60-day recency flag is confined to training mode and is never used as an agency rule.

`data/guidelines/agency/fannie_mae/` contains four original browser-print PDFs. `manifest.json` preserves source URLs, topic identifiers, verified topic revision dates, hashes, and quality notes. Some text is clipped at the right edge. The original files are retained unchanged; the app marks this evidence incomplete and links the official pages. Topic revision dates are not asserted to be effective dates for a particular loan. The supplied snapshots do not provide comprehensive agency coverage (including self-employment and multiple-job guidance).

`core.py` extracts each PDF page, keeps it under its topic identifier, and splits within page/section boundaries. The semantic option is a larger character-window baseline, not semantic boundary detection. Each chunk carries page number, source path/URL, topic revision, PDF creation timestamp, content hash, and quality warning. Manifest mismatches fail ingestion. New PDFs require extractable text, a topic heading, and an official source URL in the text; review their completeness and add a hash-bound manifest entry before relying on them.

Ollama embeddings and NumPy cosine search use local JSON indexes under `.indexes/`; optional BM25 reranking combines semantic and lexical ranks. Indexes rebuild atomically and reject changes to source content, metadata, or embedding model. No vector database or workflow server is needed. `build_graph().invoke(...)` preserves the starter interface with a synchronous retrieval/generation workflow.

The small chat model selects guideline chunk IDs through structured output. The app accepts only IDs present in the retrieved evidence and quotes their original text directly; it never displays model-written factual claims or reconstructed passages. Loan observations are rendered directly from JSON with `[loan:LOAN_ID]` citations. Selected passages carry section citations, and full retrieved excerpts and metadata are appended independently. If no valid selection is returned, the app reports insufficient evidence. This bounded, extractive answer avoids exposing the small model's unreliable free-form factual claims. It does not infer eligibility or establish that a selected passage applies to the loan. Human review is required, especially for incomplete PDF passages. Source text and questions are treated as untrusted data.

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

Tests cover synthetic records, separation of fictional rules, PDF provenance, changed-source rejection, index lifecycle, citation checks, Streamlit loan selection, cross-loan denial, stale events, and safe answer markup. `evaluate.py` measures Recall@3 against four labeled fictional-policy questions using live Ollama embeddings; the same function powers the RAG Evaluation page. `smoke_workflow.py` exercises rendering, index building, and the loan-scoped backend with live Ollama. `verify_ui.py` exercises the real browser UI, including open/close, focus, Escape, drafts, submission while minimized, citations, history restoration, loan switching, all navigation views, and mobile dimensions/keyboard behavior. Verification evidence and screenshots are saved locally under `artifacts/`.

The imported starter had misplaced contents in `requirements.txt`, `core.py`, `evaluate.py`, and `README.md`; these files were repaired. The original loan JSON, fictional policy file, and PDFs were retained. The root `test_core.py` is a misplaced duplicate loan list; pytest uses the actual tests under `tests/`. Importlib mode avoids the duplicate test module name collision.
