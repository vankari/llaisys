import json
import re
import time
import uuid
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .config import CONFIG
from .engine import ENGINE
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatResponseMessage,
    Usage,
)

app = FastAPI(title="LLAISYS Chat Server", version="0.1.0")
UI_INDEX = Path(__file__).resolve().parent / "ui" / "index.html"


@app.get("/")
def chat_ui():
    if not UI_INDEX.exists():
        raise HTTPException(status_code=404, detail="UI page not found")
    return FileResponse(UI_INDEX)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": CONFIG.device, "model": CONFIG.default_model_name}


def _chunk_payload(chunk_id: str, model_name: str, content: str, finish_reason=None):
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }


def _sanitize_completion_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned
    return text.strip()


def _longest_suffix_prefix(value: str, candidates: list[str]) -> int:
    max_len = 0
    for candidate in candidates:
        upper = min(len(value), len(candidate) - 1)
        for n in range(upper, 0, -1):
            if value.endswith(candidate[:n]):
                if n > max_len:
                    max_len = n
                break
    return max_len


class _ThinkTagStreamFilter:
    START_TAG = "<think>"
    END_TAG = "</think>"

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""

        self._buffer += chunk
        visible_parts: list[str] = []

        while self._buffer:
            if self._in_think:
                end_idx = self._buffer.find(self.END_TAG)
                if end_idx == -1:
                    keep = min(len(self._buffer), len(self.END_TAG) - 1)
                    self._buffer = self._buffer[-keep:] if keep > 0 else ""
                    break
                self._buffer = self._buffer[end_idx + len(self.END_TAG):]
                self._in_think = False
                continue

            start_idx = self._buffer.find(self.START_TAG)
            end_idx = self._buffer.find(self.END_TAG)

            if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
                visible_parts.append(self._buffer[:end_idx])
                self._buffer = self._buffer[end_idx + len(self.END_TAG):]
                continue

            if start_idx != -1:
                visible_parts.append(self._buffer[:start_idx])
                self._buffer = self._buffer[start_idx + len(self.START_TAG):]
                self._in_think = True
                continue

            keep = _longest_suffix_prefix(self._buffer, [self.START_TAG, self.END_TAG])
            if keep > 0:
                visible_parts.append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
            else:
                visible_parts.append(self._buffer)
                self._buffer = ""
            break

        return "".join(visible_parts)


def _stream_completion(req: ChatCompletionRequest, model_name: str) -> Iterator[str]:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    think_filter = _ThinkTagStreamFilter()

    for token_id in ENGINE.stream_generate(
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens or CONFIG.default_max_new_tokens,
        top_k=req.top_k if req.top_k is not None else CONFIG.default_top_k,
        top_p=req.top_p if req.top_p is not None else CONFIG.default_top_p,
        temperature=req.temperature if req.temperature is not None else CONFIG.default_temperature,
        use_cache=req.use_cache if req.use_cache is not None else CONFIG.default_use_cache,
    ):
        piece = ENGINE._tokenizer.decode([token_id], skip_special_tokens=True)
        if not piece:
            continue
        delta = think_filter.feed(piece)
        if not delta:
            continue
        payload = _chunk_payload(chunk_id, model_name, delta)
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    done_payload = _chunk_payload(chunk_id, model_name, "", finish_reason="stop")
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    model_name = req.model or CONFIG.default_model_name
    if req.stream:
        return StreamingResponse(_stream_completion(req, model_name), media_type="text/event-stream")

    result = ENGINE.generate(
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens or CONFIG.default_max_new_tokens,
        top_k=req.top_k if req.top_k is not None else CONFIG.default_top_k,
        top_p=req.top_p if req.top_p is not None else CONFIG.default_top_p,
        temperature=req.temperature if req.temperature is not None else CONFIG.default_temperature,
        use_cache=req.use_cache if req.use_cache is not None else CONFIG.default_use_cache,
    )

    completion_text = _sanitize_completion_text(result["completion_text"])

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=result["created"],
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=completion_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(result["prompt_ids"]),
            completion_tokens=len(result["completion_ids"]),
            total_tokens=len(result["prompt_ids"]) + len(result["completion_ids"]),
        ),
    )
    return response
