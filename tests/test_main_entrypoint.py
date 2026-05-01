import importlib
import runpy
import sys
import types
from pathlib import Path


def test_import_main_uses_chat_from_stubbed_module(monkeypatch):
    fake_chat_module = types.ModuleType("app.services.chat")
    fake_chat = lambda: None
    fake_chat_module.chat = fake_chat

    monkeypatch.setitem(sys.modules, "app.services.chat", fake_chat_module)
    sys.modules.pop("main", None)

    imported_main = importlib.import_module("main")

    assert imported_main.chat is fake_chat


def test_running_main_as_script_calls_chat(monkeypatch):
    calls = {"count": 0}

    fake_chat_module = types.ModuleType("app.services.chat")

    def fake_chat():
        calls["count"] += 1

    fake_chat_module.chat = fake_chat
    monkeypatch.setitem(sys.modules, "app.services.chat", fake_chat_module)

    runpy.run_path(str(Path("main.py")), run_name="__main__")

    assert calls["count"] == 1
