from typing import Sequence, Optional, Union
from pathlib import Path
import json
import ctypes

from huggingface_hub import snapshot_download

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import load_qwen2, LlaisysQwen2Meta

load_qwen2(LIB_LLAISYS)


class Qwen2:
    """Qwen2 language model implementation."""
    
    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: DeviceType = DeviceType.CPU):
        """Initialize Qwen2 model.
        
        Args:
            model_path: Path to model directory. If None, downloads default model.
            device: Device type for inference.
        """
        self.model_path = self._resolve_model_path(model_path)
        self._validate_model_files()
        
        self.device = device
        self.device_id = 0
        
        config = self._load_config()
        self._init_model_params(config)
        
        self.data_type = DataType.F32
        
        self._create_model()

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve model path, downloading if necessary."""
        if model_path is not None and Path(model_path).exists():
            return Path(model_path)
        return Path(snapshot_download(self.DEFAULT_MODEL_ID))

    def _validate_model_files(self) -> None:
        """Validate that required model files exist."""
        required_files = ["config.json"]
        for file_name in required_files:
            file_path = self.model_path / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file {file_path} not found!")

    def _load_config(self) -> dict:
        """Load model configuration from config.json."""
        config_path = self.model_path / "config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")

    def _init_model_params(self, config: dict) -> None:
        """Initialize model parameters from configuration."""
        required_params = [
            "hidden_size", "intermediate_size", "max_position_embeddings",
            "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
            "rms_norm_eps", "rope_theta", "vocab_size", "eos_token_id"
        ]
        
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required parameter: {param}")
                
        self.eos_token_id = config["eos_token_id"]
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_hidden_layers = config["num_hidden_layers"]
        self.num_key_value_heads = config["num_key_value_heads"]
        self.rms_norm_eps = config["rms_norm_eps"]
        self.rope_theta = config["rope_theta"]
        self.vocab_size = config["vocab_size"]
        
        self.per_head_dim = self.hidden_size // self.num_attention_heads

    def _create_model(self) -> None:
        """Create the model instance."""
        meta = LlaisysQwen2Meta(
            dtype=self.data_type,
            nlayer=self.num_hidden_layers,
            hs=self.hidden_size,
            nh=self.num_attention_heads,
            nkvh=self.num_key_value_heads,
            dh=self.per_head_dim,
            di=self.intermediate_size,
            maxseq=self.max_position_embeddings,
            voc=self.vocab_size,
            epsilon=self.rms_norm_eps,
            theta=self.rope_theta,
            end_token=self.eos_token_id
        )

        device_ids = (ctypes.c_int * 1)(0)
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            ctypes.c_int(self.device),
            device_ids,
            ctypes.c_int(1)
        )

        if not self.model:
            raise RuntimeError("Failed to create Qwen2 model.")

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ) -> Sequence[int]:
        """Generate tokens using the model.
        
        Args:
            inputs: Input token IDs
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter  
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs including input tokens
        """
        if not inputs:
            raise ValueError("Input tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
            
        generated = list(inputs)
        
        for _ in range(max_new_tokens):
            next_token = self._infer(generated)
            print(".")
            if next_token == self.eos_token_id:
                break
                
            generated.append(next_token)

        return generated

    def _infer(self, token_ids: Sequence[int]) -> int:
        """Perform single inference step on token sequence.
        
        Args:
            token_ids: Sequence of token IDs
            
        Returns:
            Next token ID
        """
        ntokens = len(token_ids)
        TokenArrayType = ctypes.c_int64 * ntokens
        input_array = TokenArrayType(*token_ids)

        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            input_array,
            ctypes.c_size_t(ntokens)
        )

        return next_token