"""Database adapter contracts tested offline; live comparisons use pinecone_setup.py."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import core
from ranking import rank_candidates
import vector_store as store


@pytest.fixture
def payload():
    chunks = core.split_policy('semantic', 'training')[:2]
    return {'schema':2, 'embedding_model':core.EMBED_MODEL,
        'fingerprint':core._fingerprint(chunks), 'source_signature':core._source_signature('training'),
        'chunking_version':core.CHUNKING_VERSION, 'chunks':chunks, 'vectors':[[1.,0.],[0.,1.]]}


class FakeIndex:
    def __init__(self):
        self.records = {}
        self.queries = []

    def upsert(self, namespace, vectors):
        self.records.setdefault(namespace, {}).update({v['id']:v for v in vectors})

    def fetch(self, namespace, ids):
        return SimpleNamespace(vectors={i:self.records[namespace][i] for i in ids if i in self.records[namespace]})

    def query(self, namespace, vector, top_k, include_metadata, include_values):
        self.queries.append(namespace)
        matches = []
        for record in self.records.get(namespace, {}).values():
            score = float(np.dot(record['values'], vector)/(np.linalg.norm(record['values'])*np.linalg.norm(vector)))
            matches.append(SimpleNamespace(id=record['id'],score=score))
        matches.sort(key=lambda m:(-m.score,m.id))
        return SimpleNamespace(matches=matches[:top_k])


def test_snapshot_round_trip_preserves_all_provenance_and_queries_only_its_namespace(payload, monkeypatch):
    index = FakeIndex()
    monkeypatch.setattr(store, '_index', lambda *args,**kwargs:index)
    pointer = store.upsert_snapshot(payload, 'training', 'semantic', wait_seconds=0)
    payload['pinecone'] = pointer
    records = index.records[pointer['namespace']]
    assert len(records) == len(payload['chunks'])
    for chunk in payload['chunks']:
        assert json.loads(records[chunk['id']]['metadata']['provenance_json']) == chunk
        assert 'loan_id' not in records[chunk['id']]['metadata']
    matches = store.query_snapshot(payload,'training','semantic',np.array([1.,0.]))
    assert matches[0] == (payload['chunks'][0]['id'],1.)
    assert set(index.queries) == {pointer['namespace']}


def test_namespaces_separate_corpus_strategy_and_embedding_versions(payload):
    first = store.namespace_for(payload,'training','semantic')
    assert first != store.namespace_for(payload,'fannie_mae','semantic')
    assert first != store.namespace_for(payload,'training','fixed')
    changed = {**payload,'embedding_model':'another-model'}
    assert first != store.namespace_for(changed,'training','semantic')


@pytest.mark.parametrize('failure', ['foreign-id','missing','dimension','wrong-snapshot'])
def test_invalid_remote_results_are_withheld_without_local_fallback(payload, monkeypatch, failure):
    index = FakeIndex()
    monkeypatch.setattr(store,'_index',lambda *args,**kwargs:index)
    payload['pinecone'] = store.upsert_snapshot(payload,'training','semantic',wait_seconds=0)
    namespace = payload['pinecone']['namespace']
    query = np.array([1.,0.])
    if failure == 'foreign-id':
        record = index.records[namespace].pop(payload['chunks'][0]['id'])
        record['id'] = 'another-corpus'
        index.records[namespace]['another-corpus'] = record
    elif failure == 'missing':
        index.records[namespace].clear()
    elif failure == 'dimension':
        query = np.array([1.,0.,0.])
    else:
        payload['pinecone']['namespace'] = 'other-snapshot'
    with pytest.raises(RuntimeError):
        store.query_snapshot(payload,'training','semantic',query)


def test_remote_dense_pool_keeps_bm25_expansion_without_fake_cosine_scores():
    chunks = [{'id':'dense','section_id':'A','title':'Weather','text':'Rainfall and clouds.'},
              {'id':'lexical','section_id':'B','title':'Salary','text':'Paystub employment documentation.'},
              {'id':'other','section_id':'C','title':'Weather','text':'Rainfall and clouds.'}]
    hits,trace = rank_candidates('paystub employment documentation',chunks,None,None,
        'hybrid_rerank',dense_candidates=[('dense',.8)],pool=1)
    assert hits[0]['id'] == 'lexical'
    lexical = next(r for r in trace if r['chunk_id'] == 'lexical')
    assert lexical['in_bm25_pool'] and not lexical['in_dense_pool']
    assert lexical['dense_score'] is None and lexical['dense_rank'] is None
    assert lexical['rank_after_rerank'] == 1


def test_core_uses_remote_candidates_and_preserves_local_snapshot_on_upload_failure(tmp_path,payload,monkeypatch):
    monkeypatch.setattr(core,'INDEX_DIR',tmp_path)
    path = core._index_path('semantic','training')
    path.write_text(json.dumps(payload),encoding='utf-8')
    monkeypatch.setenv('VECTOR_BACKEND','pinecone')
    monkeypatch.setattr(core,'_embed',lambda texts:np.array([[1.,0.] for _ in texts]))
    monkeypatch.setattr(core,'query_snapshot',lambda *args:[(payload['chunks'][1]['id'],.9)])
    result = core.retrieve_detailed('income','semantic','training',1,'dense')
    assert result['vector_backend'] == 'pinecone'
    assert result['hits'][0]['id'] == payload['chunks'][1]['id']
    before = path.read_bytes()
    def failed_upload(*args):
        raise RuntimeError('Upload incomplete')
    monkeypatch.setattr(core,'upsert_snapshot',failed_upload)
    with pytest.raises(RuntimeError,match='Upload incomplete'):
        core.index_policies('semantic','training')
    assert path.read_bytes() == before


def test_sdk_error_details_and_credentials_are_not_exposed():
    class Failure(Exception):
        status = 401
    def request():
        raise Failure('secret-key and Authorization header')
    with pytest.raises(RuntimeError) as caught:
        store._request(request)
    assert 'HTTP 401' in str(caught.value)
    assert 'secret-key' not in str(caught.value)


def test_absent_api_key_has_clear_local_error(monkeypatch):
    monkeypatch.delenv('PINECONE_API_KEY',raising=False)
    with pytest.raises(RuntimeError,match='PINECONE_API_KEY'):
        store._index(768)
