import importlib
import os
import sys
import tempfile


def _load_server_config_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_root = os.path.join(repo_root, "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)

    if "server.config" in sys.modules:
        del sys.modules["server.config"]

    return importlib.import_module("server.config")


def test_model_path_uses_existing_local_path(monkeypatch):
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("LLAISYS_MODEL_PATH", temp_dir)
        config_module = _load_server_config_module()
        config = config_module.ServerConfig()
        assert config.model_path == temp_dir


def test_model_path_fallbacks_to_hf_when_path_missing(monkeypatch):
    monkeypatch.setenv("LLAISYS_MODEL_PATH", "/tmp/llaisys-model-path-not-exist")
    config_module = _load_server_config_module()
    config = config_module.ServerConfig()
    assert config.model_path is None

if __name__ == "__main__":
    test_model_path_uses_existing_local_path()
    test_model_path_fallbacks_to_hf_when_path_missing()
    print("\033[92mConfig tests passed!\033[0m\n")