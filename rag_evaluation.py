"""Measured local benchmark; cached reports keep the five-minute demo responsive."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import core
from chat_service import review_request
from evaluation_metrics import answer_rubric, relevant_ids, retrieval_metrics
from ranking import MODES
from vector_store import backend_name, index_name

ROOT = Path(__file__).parent
DATASET = ROOT / 'data/evaluation/questions.json'
REPORT = ROOT / 'reports/rag_comparison.json'
ANSWER_CASES = {'A01', 'A05', 'A08', 'A12', 'T04', 'U01', 'N01', 'N03'}


def load_cases():
    return json.loads(DATASET.read_text(encoding='utf-8'))['questions']


def input_signature():
    files = [DATASET, ROOT/'core.py', ROOT/'chunking.py', ROOT/'ranking.py', ROOT/'rag_evaluation.py', ROOT/'evaluation_metrics.py', ROOT/'chat_service.py', ROOT/'corpus_scope.py', ROOT/'vector_store.py']
    data = ''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in files)
    data += core._source_signature('fannie_mae') + core._source_signature('training')
    store = json.dumps({'backend':backend_name(), 'index':index_name() if backend_name() == 'pinecone' else None,
                        'namespace_prefix':__import__('os').getenv('PINECONE_NAMESPACE_PREFIX', 'mortgage-uw')})
    return hashlib.sha256((data + core.EMBED_MODEL + core.CHAT_MODEL + core.BASE_URL + store).encode()).hexdigest()


def cached_report():
    if REPORT.exists():
        report = json.loads(REPORT.read_text(encoding='utf-8'))
        if report.get('input_signature') == input_signature():
            return report
    return None


def _mean(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _write_report(report):
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# Measured RAG comparison', '', f"Measured at {report['measured_at']} UTC; 30 labeled questions.",
        f"Models: embeddings `{core.EMBED_MODEL}`; chat `{core.CHAT_MODEL}`.", '',
        f"Vector backend: `{report.get('vector_backend', 'local')}`. BM25 and passage reranking run in Python.", '',
        '| Chunking | Retrieval | Precision@5 | Recall@5 | Median ms | P95 ms |',
        '|---|---|---:|---:|---:|---:|']
    for r in report['summary']:
        lines.append(f"| {r['strategy']} | {r['mode']} | {r['precision_at_5']:.3f} | {r['recall_at_5']:.3f} | {r['latency_median_ms']:.1f} | {r['latency_p95_ms']:.1f} |")
    lines += ['', 'Chunk-level relevance uses section IDs and prewritten source-text anchors. Precision uses five slots; missing slots count as nonrelevant. Recall excludes no-answer cases (undefined, never scored as 1). Overlap can create multiple gold chunks, and changing chunk granularity changes the recall denominator.', '',
        'Latency is one wall-clock sample per question/configuration, including index validation/loading, Ollama query embedding, and ranking. It excludes index building and chat generation. No parallel requests or query-vector cache were used. Hardware/load and first-request warmup are not controlled.', '',
        'Dense = cosine search. Hybrid = independent dense/BM25 candidate union plus reciprocal-rank fusion. Hybrid rerank = the hybrid pool followed by a deterministic query-passage lexical scorer, not a cross-encoder. The JSON contains raw scores and all before/after candidate ranks.', '', '## Answer evaluation', '']
    answers = report.get('answers', [])
    if answers:
        lines += [f"{len(answers)} actual pipeline answers measured on semantic + dense (the deployed default) only; {sum(bool(a.get('model_invoked')) for a in answers)} invoked Ollama and the remainder used documented corpus-scope abstention. Answer scores are not a six-configuration answer comparison.", '',
            '| Case | Source fidelity | Citation correctness | Support precision | Required section coverage | Correct abstention |',
            '|---|---:|---:|---:|---:|---|']
        for row in answers:
            m = row['metrics']
            coverage = 'n/a' if m['required_section_coverage'] is None else f"{m['required_section_coverage']:.2f}"
            abstain = 'n/a' if m['abstention_correct'] is None else str(m['abstention_correct'])
            lines.append(f"| {row['id']} | {m['faithfulness_source_fidelity']:.2f} | {m['citation_correctness']:.2f} | {m['answer_support_precision']:.2f} | {coverage} | {abstain} |")
    else:
        lines.append('Answers not measured in this retrieval-only run. Use `python evaluate.py --answers`; no answer scores are inferred from retrieval scores.')
    lines += ['', 'Rubric: exact quote fidelity and authoritative record equality; citation ID/section/page/version and display checks; independent gold-anchor relevance; required-section coverage; and no-answer abstention. These are automated source-fidelity checks, not human semantic entailment or policy applicability. See `evaluation_rubric.md`.', '', '## Failure analysis', '']
    failures = []
    for r in report['retrieval_results']:
        m = r['metrics']
        if m['recall_at_5'] is not None and m['recall_at_5'] < 1:
            failures.append(f"- {r['strategy']}/{r['mode']} {r['id']}: recall {m['recall_at_5']:.3f}; found {m['relevant_returned']} of {m['gold_chunk_count']} gold chunks. Top sections: {', '.join(r['sections'])}.")
    lines += failures or ['No answerable retrieval misses measured; this small curated set does not establish general reliability.']
    for r in answers:
        m = r['metrics']
        if m['answer_support_precision'] < 1 or m['abstention_correct'] is False or (m['required_section_coverage'] is not None and m['required_section_coverage'] < 1):
            lines.append(f"- Answer {r['id']}: support precision {m['answer_support_precision']:.3f}, section coverage {m['required_section_coverage']}, correct abstention {m['abstention_correct']}. Inspect the actual answer/selection in the JSON report.")
    lines += ['', '## Limitations', '',
        '- Hand-labeled, small corpus; no independent held-out set. Relevance anchors are not underwriting certifications.',
        '- Missing multiple-job, self-employment calculation, FHA, and other topics cannot be answered. Raw nearest-neighbor retrieval still returns candidates; answer-stage abstention is tested separately.',
        '- Verified snapshots do not establish effective-date applicability; see `source_quality.json`.',
        '- Semantic segmentation is deterministic TextTiling-inspired lexical cohesion, not embedding segmentation. Page/table boundaries remain limitations. Same-page exceptions attach to preceding text.',
        '- Reranking is lexical, not a learned cross-encoder. The small chat model can select irrelevant evidence or fail to abstain; exact-source fidelity can be perfect while answer relevance is poor.',
        '- Latency is single-run and load-dependent. Cached reports validate source/code/dataset hashes and model names, not silent same-name model replacement. Measured model digests are recorded.',
        '- The initial measured diagnostics are retained in `initial_rag_comparison.json` and `.md`. Subsequent source-option numbering, top-five answer context, and conservative missing-topic/ambiguity guards address observed failures; this is development-set iteration, not held-out validation. Guards describe corpus coverage, not underwriting rules. Unrecognized out-of-scope requests can still escape them.']
    (REPORT.parent/'rag_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_evaluation(force=False, include_answers=False):
    existing = None if force else cached_report()
    if existing and (not include_answers or existing.get('answers_complete')):
        return existing['summary']
    cases = load_cases()
    if existing:
        report = existing
    else:
        report = {'measured_at': datetime.now(timezone.utc).isoformat(), 'input_signature': input_signature(),
                  'dataset_version': 'week2-v1', 'summary': [], 'retrieval_results': [], 'indexes': {},
                  'vector_backend': backend_name(),
                  'model_status': core.ollama_status()}
        indexes = {}
        for strategy in ('fixed', 'semantic'):
            for corpus in ('training', 'fannie_mae'):
                started = perf_counter()
                count = core.index_policies(strategy, corpus)
                payload = json.loads(core._index_path(strategy, corpus).read_text(encoding='utf-8'))
                indexes[(strategy, corpus)] = payload['chunks']
                report['indexes'][f'{corpus}/{strategy}'] = {'chunks': count, 'build_seconds': perf_counter()-started,
                    'fingerprint': payload['fingerprint']}
        for strategy in ('fixed', 'semantic'):
            for mode in MODES:
                measured = []
                for case in cases:
                    gold = relevant_ids(case, indexes[(strategy, case['corpus'])])
                    if case['evidence'] and not gold:
                        raise ValueError(f"Gold anchors not represented: {case['id']} / {strategy}")
                    result = core.retrieve_detailed(case['question'], strategy, case['corpus'], 5, mode)
                    metrics = retrieval_metrics(result['hits'], gold)
                    row = {'id': case['id'], 'category': case['category'], 'corpus': case['corpus'],
                           'strategy': strategy, 'mode': mode, 'metrics': metrics, 'latency_ms': result['latency_ms'],
                           'hit_ids': [h['id'] for h in result['hits']], 'sections': [h['section_id'] for h in result['hits']],
                           'gold_ids': sorted(gold), 'candidate_trace': result['candidate_trace']}
                    measured.append(row)
                    report['retrieval_results'].append(row)
                latencies = sorted(r['latency_ms'] for r in measured)
                report['summary'].append({'strategy': strategy, 'mode': mode, 'cases': len(measured),
                    'precision_at_5': _mean([r['metrics']['precision_at_5'] for r in measured]),
                    'recall_at_5': _mean([r['metrics']['recall_at_5'] for r in measured]),
                    'latency_median_ms': median(latencies), 'latency_p95_ms': latencies[math.ceil(.95*len(latencies))-1]})
                print(json.dumps(report['summary'][-1]), flush=True)
        _write_report(report)
    if include_answers:
        report['answers'] = []
        for case in cases:
            if case['id'] not in ANSWER_CASES:
                continue
            context = {'loan_id': case['loan_id'], 'corpus': case['corpus'], 'strategy': 'semantic',
                       'rerank': False, 'retrieval_mode': 'dense', 'view': 'RAG Evaluation'}
            started = perf_counter()
            result = review_request(context, case['question'], [])
            chunks = json.loads(core._index_path('semantic', case['corpus']).read_text(encoding='utf-8'))['chunks']
            metrics = answer_rubric(case, result, chunks, core.get_loan(case['loan_id']))
            report['answers'].append({'id': case['id'], 'question': case['question'], 'metrics': metrics,
                'model_invoked': result.get('model_invoked'), 'generation_mode': result['generation_mode'],
                'scope_limitation': result.get('scope_limitation'),
                'answer': result['answer'], 'selected_evidence': result['selected_evidence'],
                'generation_seconds': perf_counter()-started, 'retrieval_trace': result.get('retrieval_trace', [])})
            print(json.dumps({'answer_case': case['id'], 'metrics': metrics}), flush=True)
            _write_report(report)
        report['answers_complete'] = True
        _write_report(report)
    return report['summary']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Rebuild and measure fresh retrieval results')
    parser.add_argument('--answers', action='store_true', help='Also measure eight representative live answers')
    args = parser.parse_args()
    print(json.dumps(run_evaluation(args.force, args.answers), indent=2))
