import json
from pathlib import Path

import numpy as np
import pytest

import core


def test_all_records_remain_synthetic_and_unverified():
    records = core.loans()
    assert len(records) == 5
    assert len({r['loan_id'] for r in records}) == 5
    assert all('(fictional)' in r['borrower'] and r['income_status'] == 'UNVERIFIED' for r in records)
    with pytest.raises(ValueError, match='Unknown synthetic loan'):
        core.get_loan('missing')


def test_fictional_recency_rule_never_runs_for_agency():
    record = core.get_loan('LN-10046')
    assert core.evidence_review(record)['demo_recency_flags']
    facts = core.evidence_review(record, 'fannie_mae')
    assert not facts['demo_recency_flags']
    assert facts['paystub_ages'][0]['age_days_at_application'] == 151


@pytest.mark.parametrize('strategy', ['semantic', 'fixed'])
def test_pdf_ingestion_preserves_provenance(strategy):
    chunks = core.split_policy(strategy, 'fannie_mae')
    assert {c['section_id'] for c in chunks} == {'B3-3.3-01', 'B3-3.2-01', 'B3-3.1-01', 'B3-3.3-02'}
    assert len({c['id'] for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk['source_url'].startswith('https://selling-guide.fanniemae.com/sel/')
        assert chunk['version'] in ('03/04/2026', '10/07/2026')
        assert len(chunk['sha256']) == 64
        assert chunk['page'] >= 1 and chunk['pdf_creation_date']
        assert 'clipped' in chunk['quality_warning']
        assert chunk['corpus'] == 'fannie_mae'
        assert 'TR-INC-' not in chunk['text']


def test_changed_pdf_fails_manifest_validation(tmp_path):
    original = core.AGENCY / 'base_income.pdf'
    (tmp_path / original.name).write_bytes(original.read_bytes())
    manifest = json.loads((core.AGENCY / 'manifest.json').read_text())
    manifest[original.name]['sha256'] = 'wrong'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='file changed'):
        core.split_policy('semantic', 'fannie_mae', tmp_path)


def test_index_retrieval_and_stale_model_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(core, 'INDEX_DIR', tmp_path)
    monkeypatch.setattr(core, '_embed', lambda texts: np.array([[1.0, 0.5] for _ in texts]))
    with pytest.raises(RuntimeError, match='Build'):
        core.retrieve('income')
    assert core.index_policies() >= 7
    hits = core.retrieve('bonus income', k=3)
    assert len(hits) == 3
    assert all(h['corpus'] == 'training' for h in hits)
    monkeypatch.setattr(core, 'EMBED_MODEL', 'different-model')
    with pytest.raises(RuntimeError, match='Rebuild'):
        core.retrieve('income')


def test_invented_model_evidence_is_rejected(monkeypatch):
    policies = core.split_policy()[:1]
    monkeypatch.setattr(core, 'retrieve', lambda *args: policies)
    monkeypatch.setattr(core, '_post', lambda *args: {'message': {'content': json.dumps({'chunk_ids': ['invented-source']})}})
    result = core.build_graph().invoke({'loan_id': 'LN-10045', 'question': 'income?'})
    assert not result['citation_check_passed']
    assert 'no generated claims were accepted' in result['answer']
    assert 'invented-source' not in result['answer']
    assert policies[0]['text'] in result['answer'].replace('\n  > ', '\n')


def test_streamlit_initial_render_and_loan_selection():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(core.ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    assert len(app.selectbox[0].options) == 5
    app.selectbox[0].select(app.selectbox[0].options[1]).run()
    assert not app.exception
    assert 'LN-10046' in app.subheader[0].value


def test_exact_model_quote_is_accepted_without_inventing_loan_facts(monkeypatch):
    policies = core.split_policy()[:1]
    chunk = policies[0]
    quote = chunk['text']
    monkeypatch.setattr(core, 'retrieve', lambda *args: policies)
    monkeypatch.setattr(core, '_post', lambda *args: {'message': {'content': json.dumps({
        'chunk_ids': [chunk['id']]})}})
    result = core.build_graph().invoke({'loan_id': 'LN-10046', 'question': 'Review income', 'corpus': 'training'})
    assert result['citation_check_passed']
    assert result['selected_evidence'][0]['quote'] == quote
    assert '151 days before application' in result['answer']
    assert '$6,500.00, UNVERIFIED' in result['answer']
    assert '[loan:LN-10046]' in result['answer']
