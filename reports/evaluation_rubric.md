# Evaluation rubric

Labels in `data/evaluation/questions.json` identify passages in the four official topic bodies or the explicitly fictional policies. A no-answer label means the requested rule is absent from this corpus, not absent from every guideline. Labels were authored before running the comparison.

## Retrieval

- **Gold chunk**: section ID matches a labeled section and text contains at least one labeled evidence anchor after Unicode/punctuation/whitespace normalization. Identical labels apply to both strategies. Overlap may yield several gold chunks. This is passage relevance, not every possible interpretation.
- **Precision@5**: relevant unique chunk IDs among the first five / 5. Missing result slots count against precision.
- **Recall@5**: relevant unique IDs returned / all gold IDs in that index. Undefined for no-answer cases and excluded from macro recall; never silently scored as 1.
- **Latency**: milliseconds including index validation/loading, Ollama query embedding, and ranking; excluding index construction and answer generation. One sample per question/configuration; median and P95 are descriptive, not statistically controlled.
- Candidate IDs and raw dense/BM25/RRF/passage scores, independent ranks, and before/after ranks are saved. Different score scales are not comparable probabilities.

## Answers: eight representative cases, semantic + dense only

The measured subset includes salary, documentation, multi-section guidance, an exception, multiple jobs, ambiguity, and absent-topic questions. Actual rendered answers and selections are saved, with `model_invoked` distinguishing Ollama selection from conservative corpus-scope abstention. No unexecuted answer scores are inferred. `initial_rag_comparison.*` retains the measured pre-guard failures; subsequent improvements were developed on this set, not validated on a held-out set.

1. **Faithfulness/source fidelity (0–1)**: mean of exact original-quote fidelity and structured selected-loan record equality. Empty quotes receive quote fidelity 1 because there is no unsupported quote, not because the answer was useful. This automated score is narrower than human semantic entailment.
2. **Citation correctness (0–1)**: mean of valid ID/section/positive page/revision provenance in the frozen index and display of required section/loan citations. No policy quotation means no policy-citation obligation; the selected-loan citation remains required. This is citation identity, not policy applicability.
3. **Answer support precision (0–1)**: selected IDs intersect gold relevant IDs / selected IDs. With no selected evidence, score 1 for no-answer/ambiguous cases, otherwise 0. This exposes irrelevant exact quotations.
4. **Required section coverage (0–1)**: gold sections supported by selected gold chunks / expected sections. Not applicable for absent or ambiguous topics.
5. **Correct abstention**: on no-answer/ambiguous cases, no evidence is selected and the response must show insufficient evidence. Answerable cases are excluded from this measure.

## Human review remains unscored

Policy applicability, effective dates, exception interpretation, real document adequacy, and human semantic entailment are not evaluated. Perfect source-fidelity/citation scores may coexist with irrelevant quotations or failed abstention. There is no borrower eligibility score.
