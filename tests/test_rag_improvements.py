from copy import deepcopy
import json
import re

import numpy as np
import pytest

import core
from chunking import fixed_spans, semantic_spans
from ranking import rank_candidates
from evaluation_metrics import answer_rubric, relevant_ids, retrieval_metrics
from rag_evaluation import load_cases
from corpus_scope import scope_limitation


def test_semantic_cuts_follow_a_topic_change_not_a_character_limit():
    weather = 'Rain clouds bring heavy rainfall and wet weather across the cloudy mountain region. '
    mortgage = 'Mortgage lenders review salaried income documentation and verify borrower employment records. '
    text = weather * 3 + mortgage * 3
    semantic = semantic_spans(text)
    fixed = list(fixed_spans(text, size=200, overlap=30))
    assert any(end == len(weather)*3 for _, end, _ in semantic)
    assert [(a,b) for a,b,_ in semantic] != [(a,b) for a,b,_ in fixed]
    assert ''.join(text[a:b] for a,b,_ in semantic) == text
    # Same topic words with a different length move semantic, not fixed, boundaries.
    longer = weather * 4 + mortgage * 3
    assert any(end == len(weather)*4 for _,end,_ in semantic_spans(longer))


def test_exceptions_and_alternative_paths_stay_with_the_governing_text():
    rule = 'The fictional example examines employment evidence and wage records before reviewing the reported monthly salary. '
    exception = 'However, alternative evidence may be considered when the original record is unavailable. '
    other = 'Mountain rain clouds create wet weather and rainfall across the mountain landscape. '
    text = rule + exception + '\n\n' + other * 3
    chunks = [text[a:b] for a,b,_ in semantic_spans(text)]
    assert any(rule.strip() in chunk and exception.strip() in chunk for chunk in chunks)
    assert ''.join(chunks) == text


def test_fixed_windows_cover_every_character_with_exact_overlap():
    text = ''.join(str(i % 10) for i in range(2200))
    spans = list(fixed_spans(text))
    assert spans[0][:2] == (0,800)
    assert spans[1][:2] == (650,1450)
    assert set().union(*(set(range(a,b)) for a,b,_ in spans)) == set(range(len(text)))


def test_if_condition_is_not_orphaned_from_preceding_rule():
    rule = 'This fictional sample describes income review and source documentation for a reported salary before drawing any review conclusion. '
    condition = 'If the evidence is unavailable, alternative documents need separate review. '
    other = 'Mountain weather includes rain clouds and heavy rainfall. '
    text = rule + condition + other*4
    assert any(rule in text[a:b] and condition in text[a:b] for a,b,_ in semantic_spans(text))


def test_real_corpus_strategies_have_different_boundaries_without_lost_text():
    fixed = core.split_policy('fixed', 'fannie_mae')
    semantic = core.split_policy('semantic', 'fannie_mae')
    def boundaries(chunks):
        return {(c['source_file'],c['page'],c['start_char'],c['end_char']) for c in chunks}
    assert boundaries(fixed) != boundaries(semantic)
    assert any(c['boundary_reason'] == 'lexical cohesion drop' for c in semantic)
    for chunk in semantic:
        assert chunk['section_id'] and chunk['pages'] == [chunk['page']]
        assert chunk['source_url'] and chunk['quality_warning']


def test_hybrid_union_can_find_a_candidate_outside_dense_pool_and_rerank_it():
    chunks = [{'id':f'c{i:02}', 'section_id':'X', 'title':'Background',
               'text':'Mountain clouds and rainfall.' if i < 29 else 'Paystub employment documentation.'} for i in range(30)]
    matrix = np.array([[1.0, i/30] for i in range(29)] + [[0.0,1.0]])
    query = np.array([1.0,0.0])
    dense, dense_trace = rank_candidates('paystub employment documentation', chunks, matrix, query, 'dense')
    hybrid, hybrid_trace = rank_candidates('paystub employment documentation', chunks, matrix, query, 'hybrid')
    reranked, rerank_trace = rank_candidates('paystub employment documentation', chunks, matrix, query, 'hybrid_rerank')
    assert 'c29' not in {c['chunk_id'] for c in dense_trace}
    assert 'c29' in {c['chunk_id'] for c in hybrid_trace}
    assert reranked[0]['id'] == 'c29'
    row = next(r for r in rerank_trace if r['chunk_id'] == 'c29')
    assert row['rank_after_rerank'] < row['rank_before_rerank']
    assert row['rerank_score'] is not None
    assert all('dense_score' in r and 'bm25_score' in r and 'hybrid_score' in r for r in hybrid_trace)


def test_zero_bm25_scores_cannot_add_artificial_fusion_signal():
    chunks = [{'id':f'c{i}', 'section_id':'X', 'title':'Weather', 'text':'Mountain rainfall.'} for i in range(4)]
    matrix = np.array([[1.,0.],[1.,.2],[1.,.4],[1.,.6]])
    hits, trace = rank_candidates('unmatchedword', chunks, matrix, np.array([1.,0.]), 'hybrid')
    assert all(not row['in_bm25_pool'] for row in trace)
    assert [h['id'] for h in hits] == ['c0','c1','c2','c3']


def test_metrics_do_not_turn_absent_answers_into_perfect_recall():
    metrics = retrieval_metrics([{'id':'a'},{'id':'b'}], {'a'})
    assert metrics['precision_at_5'] == .2 and metrics['recall_at_5'] == 1
    assert retrieval_metrics([{'id':'a'}], set())['recall_at_5'] is None
    cases = load_cases()
    assert len(cases) >= 20 and len({c['id'] for c in cases}) == len(cases)
    assert any(c['category'] == 'ambiguous' for c in cases)
    assert any(c['category'] == 'cross_section' for c in cases)
    assert any('multiple' in c['category'] for c in cases)


def test_rubric_exposes_irrelevant_exact_quotes_and_wrong_citations():
    record = core.get_loan('LN-10045')
    chunks = core.split_policy('fixed', 'training')
    wrong = next(c for c in chunks if c['section_id'] == 'TR-INC-06')
    case = {'evidence': {'TR-INC-01':['recent pay statement']}}
    result = {'facts': {'record': record}, 'selected_evidence': [
        {'chunk_id':wrong['id'],'section_id':wrong['section_id'],'quote':wrong['text']}],
        'answer':'[loan:LN-10045] [TR-INC-06]'}
    scores = answer_rubric(case, result, chunks, record)
    assert scores['faithfulness_source_fidelity'] == 1
    assert scores['answer_support_precision'] == 0
    result['selected_evidence'][0]['quote'] = 'Invented policy text'
    result['answer'] = 'No citations'
    damaged = answer_rubric(case, result, chunks, record)
    assert damaged['faithfulness_source_fidelity'] < 1 and damaged['citation_correctness'] < 1


def test_complete_sources_are_hash_bound_and_do_not_assert_effective_dates():
    manifest = json.loads((core.AGENCY/'manifest.json').read_text(encoding='utf-8'))
    for meta in manifest.values():
        assert meta['body_token_coverage'] == 1.0
        assert meta['effective_date'] is None
        assert meta['topic_revision_date'] == meta['version']
        assert (core.AGENCY/meta['active_file']).exists()
        assert meta['active_quality'] == 'complete_topic_body'


@pytest.mark.parametrize('question,corpus', [
    ('What is the FHA maximum debt-to-income ratio?', 'fannie_mae'),
    ('What exact second-job history is required?', 'fannie_mae'),
    ('Exactly how many months of overtime history are required?', 'training'),
    ('Is it okay?', 'fannie_mae'),
])
def test_absent_topics_and_vague_questions_do_not_reach_the_chat_model(question, corpus, monkeypatch):
    monkeypatch.setattr(core, 'retrieve', lambda *args: core.split_policy('fixed', 'training')[:1])
    def unexpected(*args):
        pytest.fail('Scope-limited requests must not ask the model to invent supporting evidence')
    monkeypatch.setattr(core, '_select_evidence', unexpected)
    result = core.build_graph().invoke({'loan_id':'LN-10045','corpus':corpus,'question':question})
    assert not result['model_invoked'] and not result['selected_evidence']
    assert 'Insufficient evidence' in result['answer']
    assert '[loan:LN-10045]' in result['answer']


def test_scope_guards_do_not_replace_real_income_documentation_questions():
    assert scope_limitation('Which documents support fixed base income?', 'fannie_mae') is None
    assert scope_limitation('What evidence is needed for bonus income?', 'training') is None


@pytest.mark.parametrize('question,expected_count', [
    ('What evidence supports the requested income?', 1),
    ('How do both documentation topics address validation?', 2),
])
def test_answer_window_preserves_retrieval_priority_and_allows_abstention(question, expected_count, monkeypatch):
    chunks = core.split_policy('semantic', 'training')[:5]
    def response(endpoint, payload):
        options = json.loads(payload['messages'][1]['content'].split('OPTIONS:\n', 1)[1])
        assert len(options) == expected_count
        assert [o['text'] for o in options] == [c['text'] for c in chunks[:expected_count]]
        return {'message': {'content': '{"chunk_ids": []}'}}
    monkeypatch.setattr(core, '_post', response)
    selected, _ = core._select_evidence(question, chunks)
    assert selected == []
