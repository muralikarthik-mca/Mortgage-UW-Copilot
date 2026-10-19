"""Pinecone dense-vector adapter; no loan records or credentials are persisted here."""
from functools import lru_cache
import hashlib
import json
import math
import os
from time import monotonic, sleep


def backend_name():
    backend = os.getenv('VECTOR_BACKEND', 'local').strip().lower()
    if backend not in ('local', 'pinecone'):
        raise RuntimeError('VECTOR_BACKEND must be local or pinecone.')
    return backend


def index_name():
    return os.getenv('PINECONE_INDEX', 'mortgage-uw-copilot').strip()


def namespace_for(payload, corpus, strategy):
    identity = json.dumps([payload['fingerprint'], payload['embedding_model'],
                           payload['source_signature'], payload['chunking_version']], sort_keys=True)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    prefix = os.getenv('PINECONE_NAMESPACE_PREFIX', 'mortgage-uw').strip()
    return f'{prefix}-{corpus}-{strategy}-{digest}'


@lru_cache(maxsize=1)
def _client(api_key):
    from pinecone import Pinecone
    return Pinecone(api_key=api_key)


def _request(call, **kwargs):
    try:
        return call(**kwargs)
    except Exception as error:
        # SDK responses may contain request details. Expose no raw headers/body.
        status = getattr(error, 'status', None) or getattr(error, 'status_code', None)
        detail = f' (HTTP {status})' if isinstance(status, int) else ''
        raise RuntimeError(f'Pinecone request failed{detail}. Check the API key, index permissions, quota, and network connection.') from None


@lru_cache(maxsize=4)
def _connect(name, dimension, api_key, create=False):
    from pinecone import ServerlessSpec
    client = _client(api_key)
    # Listing is only needed during explicit ingestion; queries never create indexes.
    if create:
        names = _request(client.list_indexes).names()
        if name not in names:
            _request(client.create_index, name=name, dimension=dimension, metric='cosine',
                     spec=ServerlessSpec(cloud=os.getenv('PINECONE_CLOUD', 'aws'),
                                         region=os.getenv('PINECONE_REGION', 'us-east-1')),
                     timeout=60)
    description = _request(client.describe_index, name=name)
    # v10 exposes legacy vector indexes under the reserved _values schema field.
    fields = getattr(getattr(description, 'schema', None), 'fields', {})
    field = fields.get('_values')
    actual_dimension = getattr(field, 'dimension', None)
    actual_metric = getattr(field, 'metric', None)
    if field is None:  # SDK 8/9 compatibility for an existing vector index.
        actual_dimension = getattr(description, 'dimension', None)
        actual_metric = getattr(description, 'metric', None)
    if actual_dimension != dimension or actual_metric != 'cosine':
        raise RuntimeError('Pinecone index must be a cosine vector index matching the Ollama embedding dimension. Choose a dedicated compatible index; existing indexes are never replaced.')
    return client.Index(host=description.host)


def _index(dimension, create=False):
    api_key = os.getenv('PINECONE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('Set PINECONE_API_KEY in the local .env file before using Pinecone.')
    return _connect(index_name(), dimension, api_key, create)


def upsert_snapshot(payload, corpus, strategy, wait_seconds=30):
    """Upload a versioned snapshot, then verify all IDs before publishing its pointer."""
    chunks, vectors = payload['chunks'], payload['vectors']
    dimension = len(vectors[0])
    index = _index(dimension, create=True)
    namespace = namespace_for(payload, corpus, strategy)
    records = []
    for chunk, vector in zip(chunks, vectors):
        # Flat indexed fields plus a lossless JSON copy retain null/list metadata.
        metadata = {key: chunk[key] for key in ('section_id','source_url','version','page','corpus','sha256','chunking_strategy')}
        metadata.update(text=chunk['text'], provenance_json=json.dumps(chunk, ensure_ascii=False),
                        embedding_model=payload['embedding_model'], fingerprint=payload['fingerprint'])
        records.append({'id': chunk['id'], 'values': vector, 'metadata': metadata})
    for start in range(0, len(records), 50):
        _request(index.upsert, namespace=namespace, vectors=records[start:start+50])
    expected = {c['id'] for c in chunks}
    deadline = monotonic() + wait_seconds
    while True:
        response = _request(index.fetch, namespace=namespace, ids=sorted(expected))
        if expected <= set(response.vectors):
            # Fetch visibility can precede search visibility. Check the query path too.
            visible = _request(index.query, namespace=namespace, vector=vectors[0],
                               top_k=len(chunks), include_metadata=False, include_values=False)
            if expected <= {m.id for m in visible.matches}:
                break
        if monotonic() >= deadline:
            raise RuntimeError('Pinecone writes are still becoming searchable. Previous snapshot is preserved; retry ingestion shortly.')
        sleep(1)
    return {'index': index_name(), 'namespace': namespace, 'dimension': dimension, 'count': len(chunks)}


def query_snapshot(payload, corpus, strategy, query, limit=12):
    pointer = payload.get('pinecone')
    expected_namespace = namespace_for(payload, corpus, strategy)
    if not pointer or pointer.get('index') != index_name() or pointer.get('namespace') != expected_namespace:
        raise RuntimeError('Build the selected Pinecone index first, or run pinecone_setup.py --migrate.')
    if len(query) != pointer['dimension']:
        raise RuntimeError('Query embedding dimension changed. Rebuild the selected index.')
    result = _request(_index(len(query)).query, namespace=expected_namespace,
                      vector=query.tolist() if hasattr(query, 'tolist') else query,
                      top_k=min(limit, len(payload['chunks'])), include_metadata=False, include_values=False)
    allowed = {c['id'] for c in payload['chunks']}
    matches = []
    seen = set()
    for match in result.matches:
        if match.id not in allowed or match.id in seen or not math.isfinite(match.score):
            raise RuntimeError('Pinecone returned an invalid or mismatched source ID; response withheld.')
        seen.add(match.id)
        matches.append((match.id, float(match.score)))
    if len(matches) != min(limit, len(allowed)):
        raise RuntimeError('Pinecone snapshot is incomplete or not searchable yet. Retry ingestion; no local fallback was used.')
    return matches
