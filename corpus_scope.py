"""Conservative scope checks about available documents, never underwriting rules."""
import re


def scope_limitation(question, corpus, has_history=False):
    text = re.sub(r'\s+', ' ', question.lower()).strip(' .?!')
    subject = re.search(r'income|pay\s?stub|pay statement|salary|salaried|bonus|overtime|document|employment|w-?2|tax|guideline|verification|loan|job|employer', text)
    if not has_history and not subject and re.search(r'\b(it|this|that|one|okay|ok)\b', text):
        return 'Please identify the income source, document, or requirement you want reviewed.'
    if corpus == 'fannie_mae':
        absent = [
            (r'\bfha\b|\bfreddie\b|\busda\b|\bva loan', 'This corpus contains only four Fannie Mae income topics, not the requested agency guidance.'),
            (r'second[- ]job|multiple[- ]jobs?', 'The multiple-jobs agency topic is not indexed.'),
            (r'(?:self[- ]employ|tax return).*(?:line number|qualifying income|calculat)|(?:line number).*(?:tax|income)', 'Detailed self-employment/tax-return calculation topics are not indexed.'),
        ]
        for pattern, reason in absent:
            if re.search(pattern, text):
                return reason
    if corpus == 'training' and re.search(r'\bovertime\b', text) and re.search(r'month|year|duration|how long', text):
        return 'The fictional training policies do not specify an overtime-history duration.'
    return None
