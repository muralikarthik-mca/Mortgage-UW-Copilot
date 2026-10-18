"""Enterprise workspace with a loan-scoped, floating Streamlit assistant."""
from html import escape
import json
from pathlib import Path

import streamlit as st

from assistant_ui import floating_assistant
from core import AGENCY, ROOT, evidence_review, get_loan, index_policies, loans, ollama_status

st.set_page_config(page_title='Mortgage UW | Review workspace', page_icon='🏦', layout='wide', initial_sidebar_state='expanded')
st.html('<style>' + (ROOT / 'ui/dashboard.css').read_text(encoding='utf-8') + '</style>')

records = loans()
with st.sidebar:
    st.html('<div class="uw-brand"><span class="uw-brand-mark">⌂</span><span>Mortgage UW<br>Copilot</span></div><div class="uw-brand-sub">Underwriting workspace</div>')
    view = st.radio('Workspace navigation', ['Dashboard', 'Loans', 'Guidelines', 'RAG Evaluation'], key='workspace_view', label_visibility='collapsed')
    st.divider()
    st.caption('ACTIVE LOAN')
    loan_id = st.selectbox('Select loan', [r['loan_id'] for r in records], key='selected_loan_id')
    st.caption(get_loan(loan_id)['employment_type'].replace('_', ' ').title() + ' · Conventional')
    st.divider()
    source = st.selectbox('Guideline source', ['Fannie Mae supplied PDFs', 'Fictional training policies'], key='source_selection')
    corpus = 'fannie_mae' if source.startswith('Fannie') else 'training'
    with st.expander('Retrieval settings'):
        strategy = st.radio('Chunking strategy', ['semantic', 'fixed'], key='strategy', help='Section/page-aware character windows; semantic uses larger windows.')
        rerank = st.checkbox('BM25 reranking', value=True, key='use_rerank')
    st.divider()
    st.html('<span class="uw-tag">SYNTHETIC TRAINING DATA</span><p class="uw-footnote">Local evidence support.<br>Final decisions require an authorized human.</p>')

record = get_loan(loan_id)
facts = evidence_review(record, corpus)
context = {'loan_id': loan_id, 'corpus': corpus, 'strategy': strategy, 'rerank': rerank, 'view': view}
received = sum(d['status'] == 'RECEIVED' for d in record['documents'])


def detail(label, value):
    st.html(f'<div class="uw-detail"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>')


def finding(title, body):
    st.html(f'<div class="uw-finding"><span class="uw-dot"></span><div><strong>{escape(title)}</strong><p>{escape(body)}</p></div></div>')


def documents():
    rows = [{'Document': d['type'].replace('_', ' ').title(), 'Date / year': str(d.get('date', d.get('year', 'Not supplied'))), 'Status': d['status'].replace('_', ' ').title()} for d in record['documents']]
    st.dataframe(rows, hide_index=True, width='stretch')


st.html(f'<div class="uw-eyebrow">Workspace / {escape(view)}</div>')
if view in ('Dashboard', 'Loans'):
    heading, status = st.columns([5, 1], vertical_alignment='center')
    with heading:
        st.title('Loan review workspace' if view == 'Dashboard' else 'Loan portfolio')
        st.html('<p class="uw-intro">A clear view of income, documentation, and the evidence still needed.</p>')
    with status:
        st.html('<span class="uw-tag">NEEDS REVIEW</span>')
    if view == 'Loans':
        with st.container(border=True, key='pipeline_card'):
            st.subheader('Synthetic loan portfolio')
            st.dataframe([{'Loan ID': r['loan_id'], 'Employment': r['employment_type'].replace('_', ' ').title(), 'Recorded monthly income': r['gross_monthly_income'], 'Income status': r['income_status']} for r in records], hide_index=True, width='stretch')
            st.caption('Choose an active loan in the left navigation to inspect its workspace.')
    st.subheader(f'{loan_id} · {record["employment_type"].replace("_", " ").title()}')
    st.caption(f'{record["program"]} / {record["agency"]} · Application {record["application_date"]}')
    cards = st.columns(4)
    cards[0].metric('Recorded monthly income', f'${record["gross_monthly_income"]:,.0f}')
    cards[1].metric('Income verification', 'Unverified')
    cards[2].metric('Documents received', f'{received} / {len(record["documents"])}')
    cards[3].metric('Paystub age at application', f'{max((p["age_days_at_application"] for p in facts["paystub_ages"]), default=0)} days' if facts['paystub_ages'] else 'Not supplied')
    st.write('')
    left, right = st.columns([1, 1.65], gap='large')
    with left:
        with st.container(border=True, key='income_card'):
            st.subheader('Income overview')
            detail('Employment type', record['employment_type'].replace('_', ' ').title())
            detail('Monthly amount', f'${record["gross_monthly_income"]:,.2f}')
            detail('Verification status', record['income_status'].title())
            detail('Application date', record['application_date'])
            detail('Agency', record['agency'])
            st.html('<p class="uw-footnote">Recorded income is not established qualifying income. Document contents are not supplied in this demo.</p>')
    with right:
        with st.container(border=True, key='documents_card'):
            st.subheader('Document inventory')
            st.caption('Inventory status from the selected synthetic loan record.')
            documents()
            st.html('<p class="uw-footnote">Received status does not establish document completeness, accuracy, or adequacy.</p>')
        with st.container(border=True, key='findings_card'):
            st.subheader('Preliminary findings')
            finding('Income remains unverified', 'Review source documents before establishing qualifying income.')
            if facts['explicitly_missing_documents']:
                finding('Outstanding inventory items', ', '.join(d.replace('_', ' ').title() for d in facts['explicitly_missing_documents']) + '. Confirm the appropriate documentation path; absence is not an automatic failure.')
            for age in facts['paystub_ages']:
                finding('Paystub timing requires context', f'Document {age["document_index"] + 1} is dated {age["age_days_at_application"]} days before application. Check the applicable guideline; no agency threshold is applied here.')
            if corpus == 'fannie_mae':
                finding('Guideline snapshot quality', 'The supplied PDFs contain clipped passages. Confirm incomplete text against the linked official source.')
            else:
                finding('Fictional policy mode', 'Demo training policies are selected. Their rules must not be attributed to an agency.')
    with st.expander('Selected loan source record'):
        st.json(record)
elif view == 'Guidelines':
    st.title('Guideline library')
    st.html('<p class="uw-intro">Traceable source documents, version metadata, and local retrieval indexes.</p>')
    st.html('<div class="uw-notice-banner">Agency snapshots and fictional training policies are indexed separately. Some PDF passages are clipped and require review of the official source.</div>')
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader(source)
        if corpus == 'fannie_mae':
            manifest = json.loads((AGENCY / 'manifest.json').read_text(encoding='utf-8'))
            for name, meta in manifest.items():
                with st.container(border=True):
                    st.markdown(f'**{meta["section_id"]}** · {name.replace("_", " ").removesuffix(".pdf").title()}')
                    st.caption(f'Topic revision {meta["version"]} · Snapshot completeness requires review')
                    st.link_button('Open official source', meta['source_url'])
                    with st.expander('Source metadata'):
                        st.json(meta)
        else:
            st.warning('FICTIONAL policies only. The 60-day training flag is not an agency rule.')
            st.markdown((ROOT / 'data/guidelines/TRAINING_ONLY_POLICIES.md').read_text(encoding='utf-8'))
    with c2:
        with st.container(border=True):
            st.subheader('Index management')
            st.caption(f'{strategy.title()} windows · BM25 {"enabled" if rerank else "disabled"}')
            if st.button('Build / rebuild guideline index', key='build_index', type='primary', width='stretch'):
                try:
                    with st.spinner('Extracting and embedding guideline evidence...'):
                        count = index_policies(strategy, corpus)
                    st.success(f'Indexed {count} chunks.')
                except Exception as exc:
                    st.error(f'Indexing failed: {exc}')
            if st.button('Check Ollama connection', key='check_ollama', width='stretch'):
                try:
                    status = ollama_status()
                    if status['missing']:
                        st.error('Missing models: ' + ', '.join(status['missing']))
                    else:
                        st.success('Chat and embedding models available.')
                    st.json(status)
                except Exception as exc:
                    st.error(f'Ollama is unavailable: {exc}')
else:
    st.title('RAG evaluation')
    st.html('<p class="uw-intro">Compare retrieval strategies against labeled fictional-policy questions.</p>')
    st.info('Four labeled questions measure Recall@3. This is a small retrieval check, not an underwriting quality or eligibility score.')
    if st.button('Run retrieval evaluation', key='run_evaluation', type='primary'):
        from evaluate import run_evaluation
        try:
            with st.spinner('Building training indexes and evaluating retrieval...'):
                st.session_state['evaluation_results'] = run_evaluation()
        except Exception as exc:
            st.error(f'Evaluation failed: {exc}')
    if st.session_state.get('evaluation_results'):
        st.dataframe(st.session_state['evaluation_results'], hide_index=True, width='stretch')
    else:
        with st.container(border=True):
            st.subheader('No evaluation run yet')
            st.caption('Run the comparison to see chunk counts and Recall@3 for semantic and fixed windows.')
    with st.expander('Evaluation method and command'):
        st.write('Section/page-aware character windows with local Ollama embeddings and BM25 reranking. No semantic boundary detector or cross-encoder is used.')
        st.code('.\\.venv314\\Scripts\\python.exe evaluate.py', language='powershell')

floating_assistant(context, record)
