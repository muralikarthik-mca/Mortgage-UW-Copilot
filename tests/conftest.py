"""Keep ordinary regression tests offline even when the demo uses Pinecone."""
import pytest


@pytest.fixture(autouse=True)
def local_vector_backend(monkeypatch):
    monkeypatch.setenv('VECTOR_BACKEND', 'local')
