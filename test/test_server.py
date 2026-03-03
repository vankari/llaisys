import importlib
import json
import os
import sys

from fastapi.testclient import TestClient


def _load_app_with_real_engine():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_root = os.path.join(repo_root, "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    if "server.app" in sys.modules:
        del sys.modules["server.app"]
    if "server.engine" in sys.modules:
        del sys.modules["server.engine"]

    return importlib.import_module("server.app")


def _collect_stream_reply(chunks):
    parts = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = item.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                parts.append(piece)
    return "".join(parts)


def test_healthz():
    app_module = _load_app_with_real_engine()
    client = TestClient(app_module.app)

    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert "model" in body


def test_chat_completion_non_stream():
    app_module = _load_app_with_real_engine()
    client = TestClient(app_module.app)

    payload = {
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
        "max_tokens": 8,
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    print(f"[non-stream reply] {body['choices'][0]['message']['content']}")
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] >= 0
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_chat_completion_stream():
    app_module = _load_app_with_real_engine()
    client = TestClient(app_module.app)

    payload = {
        "messages": [{"role": "user", "content": "你好"}],
        "stream": True,
        "max_tokens": 8,
        "top_k": 10,
        "top_p": 0.9,
        "temperature": 0.7,
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        chunks = list(resp.iter_text())

    merged = "".join(chunks)
    stream_reply = _collect_stream_reply(chunks)
    print(f"[stream reply] {stream_reply}")
    assert "chat.completion.chunk" in merged
    assert '"finish_reason": "stop"' in merged
    assert "data: [DONE]" in merged
    assert isinstance(stream_reply, str)


def test_chat_completion_empty_messages():
    app_module = _load_app_with_real_engine()
    client = TestClient(app_module.app)

    payload = {
        "messages": [],
        "stream": False,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400
    assert "messages cannot be empty" in resp.text


if __name__ == "__main__":
    test_healthz()
    test_chat_completion_non_stream()
    test_chat_completion_stream()
    test_chat_completion_empty_messages()
    print("\033[92mTest passed!\033[0m\n")
