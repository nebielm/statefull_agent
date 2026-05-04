import pytest

from app.core import settings
from app.db import vectorstores
from app.llm import client


def test_get_llm_raises_only_when_called_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", None)
    monkeypatch.setattr(client, "_llm", client._UNINITIALIZED)

    with pytest.raises(ValueError):
        client.get_llm()


def test_get_llm_caches_created_client(monkeypatch):
    calls = {"count": 0}

    class FakeChatOpenRouter:
        def __init__(self, **kwargs):
            calls["count"] += 1
            self.kwargs = kwargs

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(client, "_llm", client._UNINITIALIZED)
    monkeypatch.setattr(client, "ChatOpenRouter", FakeChatOpenRouter)

    first = client.get_llm()
    second = client.get_llm()

    assert first is second
    assert calls["count"] == 1
    assert first.kwargs["api_key"] == "test-key"
    assert first.kwargs["model"] == settings.CHAT_MODEL_NAME


def test_get_embeddings_caches_created_embeddings(monkeypatch):
    calls = {"count": 0}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls["count"] += 1
            self.kwargs = kwargs

    monkeypatch.setattr(client, "_embeddings", client._UNINITIALIZED)
    monkeypatch.setattr(client, "HuggingFaceEmbeddings", FakeEmbeddings)

    first = client.get_embeddings()
    second = client.get_embeddings()

    assert first is second
    assert calls["count"] == 1
    assert first.kwargs["model_name"] == settings.EMBEDDING_MODEL_NAME


def test_runtime_context_uses_lazy_vectorstore_getters(monkeypatch):
    core_store = object()
    user_store = object()

    monkeypatch.setattr(vectorstores, "get_core_vectorstore", lambda: core_store)
    monkeypatch.setattr(vectorstores, "get_user_memory_vectorstore", lambda: user_store)

    assert vectorstores.runtime_context() == {
        "knowledge_vectorstore": core_store,
        "user_vectorstore": user_store,
    }
