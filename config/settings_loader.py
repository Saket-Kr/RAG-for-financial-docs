import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def substitute_env_variables(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: substitute_env_variables(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_variables(item) for item in value]
    elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        default = None
        if ":" in env_var:
            env_var, default = env_var.split(":", 1)
        return os.getenv(env_var, default)
    return value


def load_settings(config_path: str = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "settings.yaml"

    config = load_yaml_config(str(config_path))
    return substitute_env_variables(config)
