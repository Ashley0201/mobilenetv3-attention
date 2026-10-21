"""模型子包。"""

from .attention import insert_eca
from .baseline import build_mobilenet_v3_large, model_summary, resolve_weights
from .eca import ECALayer, eca_kernel_size
from .registry import SUPPORTED_MODELS, build_model, describe_model, validate_model_config

__all__ = [
    "ECALayer",
    "SUPPORTED_MODELS",
    "build_mobilenet_v3_large",
    "build_model",
    "describe_model",
    "eca_kernel_size",
    "insert_eca",
    "model_summary",
    "resolve_weights",
    "validate_model_config",
]
