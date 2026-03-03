import threading
import time
from typing import List, Dict, Any, Iterator

from transformers import AutoTokenizer
import llaisys

from .config import CONFIG


class ChatEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        tokenizer_source = CONFIG.model_path or llaisys.models.Qwen2.DEFAULT_MODEL_ID
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        self._model = llaisys.models.Qwen2(
            model_path=CONFIG.model_path,
            device=llaisys.DeviceType.CPU if CONFIG.device == "cpu" else llaisys.DeviceType.NVIDIA,
        )

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate(self, messages: List[Dict[str, str]], max_tokens: int, top_k: int, top_p: float, temperature: float, use_cache: bool) -> Dict[str, Any]:
        prompt = self._build_prompt(messages)
        input_ids = self._tokenizer.encode(prompt)

        with self._lock:
            output_ids = self._model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
            )

        completion_ids = output_ids[len(input_ids):]
        completion_text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)

        return {
            "prompt_ids": input_ids,
            "completion_ids": completion_ids,
            "completion_text": completion_text,
            "created": int(time.time()),
        }

    def stream_generate(self, messages: List[Dict[str, str]], max_tokens: int, top_k: int, top_p: float, temperature: float, use_cache: bool) -> Iterator[int]:
        prompt = self._build_prompt(messages)
        input_ids = self._tokenizer.encode(prompt)
        generated = list(input_ids)

        with self._lock:
            for _ in range(max_tokens):
                output_ids = self._model.generate(
                    generated,
                    max_new_tokens=1,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                )

                if len(output_ids) <= len(generated):
                    break

                token_id = output_ids[len(generated)]
                generated = output_ids
                yield token_id

                if token_id == self._model.eos_token_id:
                    break


ENGINE = ChatEngine()
