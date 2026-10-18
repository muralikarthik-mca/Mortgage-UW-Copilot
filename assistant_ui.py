"""Streamlit bridge for the isolated floating-assistant assets."""
import html
from pathlib import Path
import re
from urllib.parse import urlparse

import streamlit as st

from chat_service import EXECUTOR, context_id, review_request, validate_event

ASSETS = Path(__file__).parent / 'ui'
assistant_component = st.components.v2.component(
    'underwriting_assistant', html=(ASSETS / 'assistant.html').read_text(encoding='utf-8'),
    css=(ASSETS / 'assistant.css').read_text(encoding='utf-8'),
    js=(ASSETS / 'assistant.js').read_text(encoding='utf-8'))

PROMPTS = ['Review the income documentation and outstanding items.',
           'How old is the paystub relative to the application date?',
           'What remains unverified about this income?']


def answer_html(text):
    """Restricted Markdown: escape all input before adding our own safe tags."""
    marker = '**Retrieved source evidence**'
    if marker in text:
        main, sources = text.split(marker, 1)
        return answer_html(main) + '<details><summary>Sources and full retrieved evidence</summary>' + answer_html(sources) + '</details>'
    escaped = html.escape(text)
    def link(match):
        label, url = match.groups()
        parsed = urlparse(html.unescape(url))
        if parsed.scheme == 'https' and (parsed.hostname == 'fanniemae.com' or (parsed.hostname or '').endswith('.fanniemae.com')):
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
        return label
    escaped = re.sub(r'\[([^\]\n]+)\]\(([^\s)]+)\)', link, escaped)
    escaped = re.sub(r'\*\*([^*\n]+)\*\*', r'<strong>\1</strong>', escaped)
    lines = []
    for line in escaped.splitlines():
        if not line.strip():
            continue
        if line.startswith('&gt; '):
            lines.append('<blockquote>' + line[5:] + '</blockquote>')
        else:
            lines.append('<p>' + line + '</p>')
    return ''.join(lines)


@st.fragment(run_every='1s')
def floating_assistant(context, record):
    scope = context_id(context)
    histories = st.session_state.setdefault('assistant_histories', {})
    jobs = st.session_state.setdefault('assistant_jobs', {})
    seen = st.session_state.setdefault('assistant_seen_requests', set())
    messages = histories.setdefault(scope, [])
    # Collect each job into its original context, even if the user changed loans.
    for job_scope, future in list(jobs.items()):
        if future.done():
            try:
                result = future.result()
                state = 'insufficient' if 'Insufficient evidence' in result['answer'] else None
                display = result['answer']
                for policy in {p['section_id']: p for p in result['policies']}.values():
                    if policy['source_url']:
                        display = display.replace(f"[{policy['section_id']}]", f"[{policy['section_id']}]({policy['source_url']})")
                histories[job_scope].append({'role': 'assistant', 'text': result['answer'],
                    'html': answer_html(display), 'state': state,
                    'audit': {'loan_facts': result['facts'], 'retrieved_chunks': result['policies'],
                              'citation_check_passed': result['citation_check_passed']}})
                st.session_state.setdefault('assistant_results', {})[job_scope] = result
            except Exception as exc:
                message = str(exc)
                histories[job_scope].append({'role': 'assistant', 'text': message,
                                            'html': answer_html(message), 'state': 'error'})
            del jobs[job_scope]
    widget_state = st.session_state.get('floating_assistant', {})
    is_open = widget_state.get('is_open', False) if hasattr(widget_state, 'get') else False
    event = assistant_component(key='floating_assistant', data={
        'context_id': scope, 'loan_id': context['loan_id'], 'employment_type': record['employment_type'],
        'source_label': 'Fannie Mae PDF snapshots' if context['corpus'] == 'fannie_mae' else 'Fictional training policies',
        'view': context['view'], 'messages': messages, 'busy': scope in jobs,
        'prompts': PROMPTS, 'is_open': is_open},
        on_submit_change=lambda: None, on_is_open_change=lambda: None, height=0)
    submission = event.submit
    if submission and submission.get('request_id') not in seen:
        request_id = submission.get('request_id')
        if request_id:
            seen.add(request_id)
        try:
            question = validate_event(submission, context)
            if scope in jobs:
                raise ValueError('A review is already running for this loan. Please wait for it to finish.')
            history = [{'role': m['role'], 'content': m['text']} for m in messages if m.get('state') != 'error']
            messages.append({'role': 'user', 'text': question})
            jobs[scope] = EXECUTOR.submit(review_request, dict(context), question, history)
        except ValueError as exc:
            messages.append({'role': 'assistant', 'text': str(exc), 'html': answer_html(str(exc)), 'state': 'error'})
        st.rerun(scope='fragment')
