"""Deterministic, inspectable metrics; no LLM judge or invented evaluation scores."""
import re
import unicodedata


def normalized(text):
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(re.findall(r'[a-z0-9]+', text))


def relevant_ids(case, chunks):
    evidence = case['evidence']
    return {c['id'] for c in chunks if c['section_id'] in evidence and any(
        normalized(anchor) in normalized(c['text']) for anchor in evidence[c['section_id']])}


def retrieval_metrics(hits, gold_ids, k=5):
    found = {h['id'] for h in hits[:k]}
    relevant = len(found & gold_ids)
    return {'precision_at_5': relevant / k,
            'recall_at_5': relevant / len(gold_ids) if gold_ids else None,
            'relevant_returned': relevant, 'gold_chunk_count': len(gold_ids),
            'unanswerable': not bool(gold_ids)}


def answer_rubric(case, result, chunks, authoritative_record):
    """Source fidelity != applicability. The report keeps these measures separate.

    Faithfulness = mean(exact quote fidelity, exact loan-record fidelity).
    Citation correctness = mean(valid quote/section/page/source provenance,
    displayed section/loan references). Semantic relevance uses independent gold.
    """
    by_id = {c['id']: c for c in chunks}
    selected = result['selected_evidence']
    faithful = [item['chunk_id'] in by_id and item['quote'] == by_id[item['chunk_id']]['text'] for item in selected]
    provenance = [item['chunk_id'] in by_id and item['section_id'] == by_id[item['chunk_id']]['section_id']
                  and by_id[item['chunk_id']]['page'] >= 1 and bool(by_id[item['chunk_id']]['version'])
                  for item in selected]
    displayed = [f"[{item['section_id']}]" in result['answer'] for item in selected]
    loan_fidelity = result['facts']['record'] == authoritative_record
    loan_cited = f"[loan:{authoritative_record['loan_id']}]" in result['answer']
    quote_score = sum(faithful)/len(faithful) if faithful else 1.0
    cite_provenance = sum(provenance)/len(provenance) if provenance else 1.0
    displayed_score = (sum(displayed)+int(loan_cited))/(len(displayed)+1)
    gold = relevant_ids(case, chunks)
    chosen = {item['chunk_id'] for item in selected}
    support = len(chosen & gold)/len(chosen) if chosen else (0.0 if gold else 1.0)
    expected_sections = set(case['evidence'])
    chosen_sections = {item['section_id'] for item in selected if item['chunk_id'] in gold}
    coverage = len(chosen_sections & expected_sections)/len(expected_sections) if expected_sections else None
    return {'faithfulness_source_fidelity': (quote_score+int(loan_fidelity))/2,
            'citation_correctness': (cite_provenance+displayed_score)/2,
            'answer_support_precision': support, 'required_section_coverage': coverage,
            'abstention_correct': (not bool(selected) and 'Insufficient evidence' in result['answer']) if not gold else None,
            'exact_quote_fidelity': quote_score, 'loan_record_fidelity': int(loan_fidelity),
            'selected_ids': sorted(chosen),
            'rubric_scope': 'Automated exact-source/record fidelity and identifier provenance; not a human applicability judgment.'}
