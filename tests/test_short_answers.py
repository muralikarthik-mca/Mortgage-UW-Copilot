from copy import deepcopy
import pytest
import core


@pytest.mark.parametrize('question', [
    'How old is the paystub?',
    'How old is the paystub relative to the application date?',
    'What is the age of the pay statement?',
    'Paystub age',
])
def test_age_question_is_short_and_uses_record_without_models(question, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('A simple loan date calculation must not call guideline retrieval or Ollama')
    monkeypatch.setattr(core, 'retrieve', unexpected)
    monkeypatch.setattr(core, '_select_evidence', unexpected)
    result = core.build_graph().invoke({'loan_id': 'LN-10046', 'question': question, 'corpus': 'fannie_mae'})
    assert '**151 days old as of the application date**' in result['answer']
    assert '2026-05-12 to 2026-10-10' in result['answer']
    assert '[loan:LN-10046]' in result['answer']
    assert len(result['answer'].split()) < 35
    assert 'Insufficient evidence' not in result['answer']
    assert result['policies'] == []


@pytest.mark.parametrize('question', [
    'Is the paystub too old under Fannie Mae guidelines?',
    'How old is the paystub and is it acceptable?',
    'Give a detailed review of paystub age and documentation requirements.',
])
def test_policy_questions_keep_the_guideline_workflow(question):
    record = core.get_loan('LN-10046')
    assert core.paystub_age_answer(question, record, core.evidence_review(record)) is None


def test_multiple_paystubs_and_no_paystub():
    record = core.get_loan('LN-10048')
    text = core.paystub_age_answer('How old are the paystubs?', record, core.evidence_review(record))
    assert 'paystub primary' in text and '**15 days' in text
    assert 'paystub secondary' in text and '**20 days' in text
    record = core.get_loan('LN-10049')
    assert 'no dated paystub' in core.paystub_age_answer('Paystub age', record, core.evidence_review(record))


def test_date_after_application_is_not_reported_as_negative_age():
    record = deepcopy(core.get_loan('LN-10045'))
    record['documents'][0]['date'] = '2026-10-15'
    text = core.paystub_age_answer('Paystub age', record, core.evidence_review(record))
    assert '**5 days after the application date**' in text
