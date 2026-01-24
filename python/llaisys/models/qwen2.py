from typing import Sequence, Optional, Union
from pathlib import Path
import json
import ctypes

from huggingface_hub import snapshot_download
import numpy as np
import torch

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import load_qwen2, LlaisysQwen2Meta
from ..tensor import Tensor

load_qwen2(LIB_LLAISYS)

class Qwen2:
    """Qwen2 language model implementation."""
    
    DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    def __init__(self, model_path: Optional[Union[str, Path]] = None, device: DeviceType = DeviceType.CPU):
        """Initialize Qwen2 model.
        
        Args:
            model_path: Path to model directory. If None, downloads default model.
            device: Device type for inference.
            
        Raises:
            ValueError: If unsupported device is specified.
            FileNotFoundError: If required model files are missing.
        """
        if device != DeviceType.CPU:
            raise ValueError("Only CPU device is currently supported")
            
        self.model_path = self._resolve_model_path(model_path)
        self._validate_model_files()
        
        self.device = device
        self.device_id = 0
        
        config = self._load_config()
        self._init_model_params(config)
        
        self.data_type = DataType.F32
        
        self._create_model()
        self._load_weights()

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
        # Required parameters
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
        
        # Derived parameters
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

    def _load_weights(self) -> None:
        """Load model weights from HuggingFace model."""
        print(f"Loading weights from {self.model_path}")
        
        weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights:
            raise RuntimeError("Failed to get Qwen2 weights.")
        
        # Load using transformers library
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            state_dict = model.state_dict()
            
            # Load embedding
            self._load_weight(weights, state_dict, "model.embed_tokens.weight", "in_embed")
            self._load_weight(weights, state_dict, "lm_head.weight", "out_embed")
            self._load_weight(weights, state_dict, "model.norm.weight", "out_norm_w")
            
            # Load layers
            for layer_idx in range(self.num_hidden_layers):
                self._load_layer_weights(weights, state_dict, layer_idx)
                
            print("All weights loaded successfully!")
            del model
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise

    def _load_weight(self, weights, state_dict, weight_key, field_name):
        """Load a single weight tensor."""
        if weight_key not in state_dict:
            print(f"Warning: {weight_key} not found in state_dict")
            return
        
        tensor_data = state_dict[weight_key].cpu().numpy().astype(np.float32)
        tensor = Tensor(shape=tensor_data.shape, dtype=DataType.F32, device=self.device, device_id=self.device_id)
        tensor.load(tensor_data)
        
        setattr(weights.contents, field_name, tensor._tensor)

    def _load_layer_weights(self, weights, state_dict, layer_idx: int):
        """Load all weights for a single layer."""
        layer_prefix = f"model.layers.{layer_idx}"
        
        weight_map = [
            ("input_layernorm.weight", "attn_norm_w"),
            ("self_attn.q_proj.weight", "attn_q_w"),
            ("self_attn.q_proj.bias", "attn_q_b"),
            ("self_attn.k_proj.weight", "attn_k_w"),
            ("self_attn.k_proj.bias", "attn_k_b"),
            ("self_attn.v_proj.weight", "attn_v_w"),
            ("self_attn.v_proj.bias", "attn_v_b"),
            ("self_attn.o_proj.weight", "attn_o_w"),
            ("post_attention_layernorm.weight", "mlp_norm_w"),
            ("mlp.gate_proj.weight", "mlp_gate_w"),
            ("mlp.up_proj.weight", "mlp_up_w"),
            ("mlp.down_proj.weight", "mlp_down_w"),
        ]
        
        for weight_name, field_name in weight_map:
            full_key = f"{layer_prefix}.{weight_name}"
            if full_key not in state_dict:
                print(f"Warning: {full_key} not found")
                continue
            
            tensor_data = state_dict[full_key].cpu().numpy().astype(np.float32)
            tensor = Tensor(shape=tensor_data.shape, dtype=DataType.F32, device=self.device, device_id=self.device_id)
            tensor.load(tensor_data)
            
            # Get the array pointer and set the element
            arr_ptr = getattr(weights.contents, field_name)
            arr_ptr[layer_idx] = tensor._tensor

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
            top_k: Top-k sampling parameter (not implemented)
            top_p: Top-p sampling parameter (not implemented)
            temperature: Sampling temperature (not implemented)
            
        Returns:
            Generated token IDs including input tokens
        """
        if not inputs:
            raise ValueError("Input tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
            
        generated = list(inputs)
        
        for step in range(max_new_tokens):
            # Prepare input array
            input_array = (ctypes.c_int64 * len(generated))(*generated)
            
            # Call inference
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                self.model,
                input_array,
                ctypes.c_size_t(len(generated))
            )
            
            if next_token < 0 or next_token == self.eos_token_id:
                break
            
            generated.append(int(next_token))
        
        return generated

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model:
            try:
                LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            except Exception:
                pass