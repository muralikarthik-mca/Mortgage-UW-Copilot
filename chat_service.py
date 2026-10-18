"""Loan-scoped assistant requests; frontend events never choose a loan record."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re

from core import build_graph, get_loan

EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='loan-review')


def context_id(context):
    fields = {key: context[key] for key in ('loan_id', 'corpus', 'strategy', 'rerank')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:24]


def review_request(context, question, history, workflow=None):
    """Resolve all facts from trusted server context, reject explicit cross-loan requests."""
    question = question.strip()
    if not question or len(question) > 2000:
        raise ValueError('Enter a question between 1 and 2,000 characters.')
    selected = context['loan_id']
    mentioned = {f'LN-{m}' for m in re.findall(r'\bLN[\s_-]*(\d+)\b', question, re.I)}
    if mentioned - {selected}:
        raise ValueError(f'This assistant can only review {selected}. Select a different loan in the workspace to review it.')
    record = get_loan(selected)
    # Only same-context history is retained by the server; client history is never accepted.
    prior_questions = [m['content'] for m in history[-6:] if m['role'] == 'user'][-2:]
    retrieval_question = '\n'.join(prior_questions + [question, 'Income type: ' + record['employment_type']])
    request = {**context, 'question': question, 'retrieval_question': retrieval_question,
               'underwriting_context': {'view': context.get('view', 'Dashboard'),
                   'employment_type': record['employment_type'], 'income_status': record['income_status'],
                   'application_date': record['application_date']},
               'conversation_history': history[-6:]}
    result = (workflow or build_graph()).invoke(request)
    if result['facts']['record']['loan_id'] != selected or result.get('loan_id') != selected:
        raise RuntimeError('The review returned a mismatched loan context and was withheld.')
    return result


def validate_event(event, context):
    if not isinstance(event, dict) or event.get('context_id') != context_id(context):
        raise ValueError('The loan context changed. Submit your question again for the selected loan.')
    if not isinstance(event.get('text'), str) or not isinstance(event.get('request_id'), str):
        raise ValueError('Invalid assistant request.')
    return event['text']
