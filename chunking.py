"""Lossless fixed windows and TextTiling-inspired lexical topic segmentation.

Semantic boundaries use TF-IDF lexical cohesion between neighboring sentence
windows, plus paragraph boundaries. No model calls are needed during splitting.
Exceptions/alternatives stay attached to their preceding governing text.
"""
import re
from collections import Counter
import math

CHUNKING_VERSION = 'lexical-cohesion-v2'
STOP = frozenset('a an the is are was were be been to of in on for and or as by with this that it its from must should may can have has will'.split())
EXCEPTION = re.compile(r'^\s*(?:however\b|unless\b|except\b|if\b|but\b|provided\b|alternativ\w*\b|note\b|otherwise\b|additionally\b|for example\b|when a documented\b)', re.I)


def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+(?:[.-][a-z0-9]+)*', text.lower()) if t not in STOP]


def fixed_spans(text, size=800, overlap=150):
    if size <= overlap or overlap < 0:
        raise ValueError('Chunk size must exceed nonnegative overlap')
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        if text[start:end].strip():
            yield start, end, {'boundary_reason': 'fixed character window'}
        if end == len(text):
            break


def _sentence_spans(text):
    # Preserve offsets and original text. Paragraph breaks are semantic candidates,
    # not instructions to discard or paraphrase text.
    start = 0
    for match in re.finditer(r'(?<=[.!?])\s+(?=[A-Z(\d])|\n\s*\n', text):
        if text[start:match.end()].strip():
            yield start, match.end()
        start = match.end()
    if text[start:].strip():
        yield start, len(text)


def semantic_spans(text, minimum_chars=120):
    units = list(_sentence_spans(text))
    if not units:
        return []
    documents = [Counter(tokens(text[a:b])) for a, b in units]
    df = Counter(term for document in documents for term in document)
    idf = {term: math.log((1 + len(documents)) / (1 + count)) + 1 for term, count in df.items()}

    def vector(lo, hi):
        counts = sum(documents[lo:hi], Counter())
        return {term: count * idf[term] for term, count in counts.items()}

    def cosine(left, right):
        numerator = sum(value * right.get(term, 0) for term, value in left.items())
        denominator = math.sqrt(sum(x*x for x in left.values()) * sum(x*x for x in right.values()))
        return numerator / denominator if denominator else 0.0

    cohesion = [cosine(vector(max(0, i-2), i), vector(i, min(len(units), i+2))) for i in range(1, len(units))]
    cuts = [0]
    reasons = {}
    for i, score in enumerate(cohesion, 1):
        current = text[units[i][0]:units[i][1]].strip()
        previous = text[units[i-1][0]:units[i-1][1]].rstrip()
        # Do not orphan a note, exception, alternate path, or a list following ':' .
        if EXCEPTION.match(current) or previous.endswith(':'):
            continue
        paragraph = bool(re.search(r'\n\s*\n\s*$', text[units[i-1][0]:units[i][0]]))
        neighbors = cohesion[max(0, i-2):min(len(cohesion), i+1)]
        valley = score <= min(neighbors) and score < 0.32
        topic_change = score < 0.12 or (valley and (max(neighbors) - score >= 0.08))
        if units[i][0] - units[cuts[-1]][0] >= minimum_chars and (topic_change or (paragraph and score < 0.45)):
            cuts.append(i)
            reasons[i] = {'boundary_reason': 'lexical cohesion drop' if topic_change else 'paragraph topic boundary',
                          'boundary_cohesion': round(score, 6)}
    cuts.append(len(units))
    return [(units[a][0], units[b-1][1], reasons.get(a, {'boundary_reason': 'source page/section start'}))
            for a, b in zip(cuts, cuts[1:])]
