"""Auditable dense/hybrid retrieval and a deterministic query-passage reranker."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from chunking import tokens

MODES = ('dense', 'hybrid', 'hybrid_rerank')


def passage_score(question, text):
    """Late query/passage interaction: best sentence coverage + phrase matches.

    This is an interpretable lexical reranker, not a learned cross-encoder.
    Its score is not a probability of policy applicability.
    """
    query = tokens(question)
    terms = set(query)
    if not terms:
        return 0.0
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    coverage = max((len(terms & set(tokens(s))) / len(terms) for s in sentences), default=0)
    pairs = set(zip(query, query[1:]))
    body = tokens(text)
    phrase = len(pairs & set(zip(body, body[1:]))) / max(len(pairs), 1)
    total = len(terms & set(body)) / len(terms)
    return 0.55 * coverage + 0.30 * phrase + 0.15 * total


def rank_candidates(question, chunks, matrix, query, mode='hybrid_rerank', k=5, pool=12, dense_candidates=None):
    if mode not in MODES or k < 1:
        raise ValueError('Unsupported retrieval mode or nonpositive k')
    if dense_candidates is None:
        dense = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query))
        dense_order = sorted(range(len(chunks)), key=lambda i: (-float(dense[i]), chunks[i]['id']))
    else:
        # Real database search scores; do not recompute NumPy nearest neighbors.
        positions = {c['id']: i for i,c in enumerate(chunks)}
        dense = np.full(len(chunks), -np.inf)
        for chunk_id, score in dense_candidates:
            if chunk_id not in positions:
                raise ValueError('Dense candidate is outside the selected source snapshot')
            dense[positions[chunk_id]] = score
        dense_order = sorted((positions[cid] for cid,_ in dense_candidates), key=lambda i: (-float(dense[i]), chunks[i]['id']))
    lexical = (BM25Okapi([tokens(c['title'] + ' ' + c['text']) for c in chunks]).get_scores(tokens(question))
               if mode != 'dense' else np.zeros(len(chunks)))
    lexical_order = sorted(range(len(chunks)), key=lambda i: (-float(lexical[i]), chunks[i]['id']))
    dr = {i: rank+1 for rank, i in enumerate(dense_order)}
    br = {i: rank+1 for rank, i in enumerate(lexical_order)}
    limit = max(pool, k)
    # Hybrid retrieval is an independent union: BM25 may introduce candidates
    # outside the dense pool. It is not just BM25 over dense-only candidates.
    dense_ids = set(dense_order[:limit])
    bm25_ids = {i for i in lexical_order[:limit] if lexical[i] > 0}
    ids = set(dense_ids)
    if mode != 'dense':
        ids.update(bm25_ids)
    rrf = {i: (1/(60+dr[i]) if i in dense_ids else 0)
              + (1/(60+br[i]) if i in bm25_ids else 0) for i in ids}
    before = sorted(ids, key=lambda i: (-float(dense[i]), chunks[i]['id'])) if mode == 'dense' else sorted(ids, key=lambda i: (-rrf[i], chunks[i]['id']))
    reranked = {i: passage_score(question, chunks[i]['text']) for i in before} if mode == 'hybrid_rerank' else {}
    after = sorted(before, key=lambda i: (-reranked[i], -rrf[i], chunks[i]['id'])) if reranked else before
    before_rank = {i: rank+1 for rank, i in enumerate(before)}
    final_rank = {i: rank+1 for rank, i in enumerate(after)}
    trace = [{'chunk_id': chunks[i]['id'], 'section_id': chunks[i]['section_id'],
              'dense_score': float(dense[i]) if np.isfinite(dense[i]) else None, 'dense_rank': dr.get(i),
              'in_dense_pool': i in dense_ids, 'in_bm25_pool': i in bm25_ids if mode != 'dense' else False,
              'bm25_score': float(lexical[i]) if mode != 'dense' else None,
              'bm25_rank': br[i] if mode != 'dense' else None,
              'hybrid_score': rrf[i] if mode != 'dense' else None,
              'rank_before_rerank': before_rank[i], 'rerank_score': reranked.get(i),
              'rank_after_rerank': final_rank[i], 'selected': final_rank[i] <= k}
             for i in before]
    by_index = {i: row for i, row in zip(before, trace)}
    hits = [{**chunks[i], 'similarity': float(dense[i]) if np.isfinite(dense[i]) else None, 'ranking': by_index[i]} for i in after[:k]]
    return hits, trace
