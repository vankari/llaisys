import json
import time
import uuid
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

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


def _stream_completion(req: ChatCompletionRequest, model_name: str) -> Iterator[str]:
    result = ENGINE.generate(
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens or CONFIG.default_max_new_tokens,
        top_k=req.top_k if req.top_k is not None else CONFIG.default_top_k,
        top_p=req.top_p if req.top_p is not None else CONFIG.default_top_p,
        temperature=req.temperature if req.temperature is not None else CONFIG.default_temperature,
        use_cache=req.use_cache if req.use_cache is not None else CONFIG.default_use_cache,
    )

    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    for token_id in result["completion_ids"]:
        piece = ENGINE._tokenizer.decode([token_id], skip_special_tokens=True)
        if not piece:
            continue
        payload = _chunk_payload(chunk_id, model_name, piece)
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

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    response = ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=result["created"],
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatResponseMessage(role="assistant", content=result["completion_text"]),
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
