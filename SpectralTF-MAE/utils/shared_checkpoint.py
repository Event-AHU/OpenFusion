"""Shared dual-spectrum peak-aware checkpoint contract for all SpectralTF downstream tasks.

A fine-tuning task must not silently reinitialize mismatched pretrained layers.
The two spectral directions load their matching input adapter/decoder plus the full
encoder; the ne/Ti/Te regressor loads the same pretrained XCS input adapter and
encoder and initializes only its three task-specific heads.
"""

from pathlib import Path
from typing import Mapping

import torch

SHARED_INPUT_SIZES = {"xcs": (35, 490), "cxrs": (12, 1024)}
SHARED_PATCH_SIZES = {"xcs": (5, 14), "cxrs": (3, 16)}


def validate_shared_geometry(domain_specs):
    """Reject any accidental divergence between pretrain and downstream geometry."""
    for domain, spec in domain_specs.items():
        if domain not in SHARED_PATCH_SIZES:
            raise ValueError(f"Unexpected SpectralTF domain: {domain!r}")
        patch = tuple(map(int, spec["patch_size"]))
        size = tuple(map(int, spec["input_size"]))
        if patch != SHARED_PATCH_SIZES[domain] or size != SHARED_INPUT_SIZES[domain]:
            raise ValueError(
                f"{domain} geometry must match the shared peak-aware checkpoint: "
                f"input_size={SHARED_INPUT_SIZES[domain]}, patch_size={SHARED_PATCH_SIZES[domain]}; "
                f"got input_size={size}, patch_size={patch}. Re-run joint pretraining "
                "before changing the common geometry."
            )


def _as_domains(value):
    if isinstance(value, str):
        return set(x for x in value.split("-") if x)
    if isinstance(value, (list, tuple)):
        return set(value)
    return set()


def _as_args_dict(checkpoint):
    source_args = checkpoint.get("args", None)
    if isinstance(source_args, Mapping):
        return source_args
    if hasattr(source_args, "__dict__"):
        return vars(source_args)
    raise ValueError(
        "Shared pretraining checkpoint is missing its training 'args' metadata. "
        "Cannot prove that it was trained with the required peak-aware mask. "
        "Use checkpoint_best.pth produced by train_SpectralTF.py."
    )


def read_verified_pretraining_checkpoint(path, device="cpu", expected_model_name=None):
    """Load and validate provenance, masking, domains, geometry and token count."""
    if not path:
        raise ValueError("A shared joint peak-aware pretraining checkpoint is required")
    if not Path(path).is_file():
        raise FileNotFoundError(f"Shared pretraining checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Shared pretraining checkpoint must be a dictionary with model and args")
    metadata = _as_args_dict(checkpoint)
    if metadata.get("spectral_masking") != "peak_aware":
        raise ValueError("Shared pretraining checkpoint must have spectral_masking=peak_aware")
    if _as_domains(metadata.get("in_domains")) != {"xcs", "cxrs"} or _as_domains(metadata.get("out_domains")) != {"xcs", "cxrs"}:
        raise ValueError("Shared pretraining checkpoint must be trained jointly on xcs and cxrs")
    if expected_model_name is not None and metadata.get("model") != expected_model_name:
        raise ValueError(
            f"Shared checkpoint model={metadata.get('model')!r} != "
            f"downstream model={expected_model_name!r}"
        )
    for domain in ("cxrs", "xcs"):
        patch = tuple(map(int, metadata.get(f"{domain}_patch_size", ())))
        size = tuple(map(int, metadata.get(f"{domain}_input_size", ())))
        if domain == "xcs" and metadata.get("pad_xcs_width_to") is not None:
            size = (size[0], int(metadata["pad_xcs_width_to"])) if len(size) == 2 else size
        if patch != SHARED_PATCH_SIZES[domain] or size != SHARED_INPUT_SIZES[domain]:
            raise ValueError(
                f"Shared checkpoint has incompatible {domain} geometry "
                f"(size={size}, patch={patch}); required "
                f"(size={SHARED_INPUT_SIZES[domain]}, patch={SHARED_PATCH_SIZES[domain]})"
            )
    total_tokens = sum(
        (SHARED_INPUT_SIZES[domain][0] // SHARED_PATCH_SIZES[domain][0])
        * (SHARED_INPUT_SIZES[domain][1] // SHARED_PATCH_SIZES[domain][1])
        for domain in ("cxrs", "xcs")
    )
    keep = int(metadata.get("num_encoded_tokens", -1))
    if not 0 < keep < total_tokens:
        raise ValueError(
            f"Pretraining checkpoint does not represent effective masking: "
            f"num_encoded_tokens={keep}, total_tokens={total_tokens}"
        )
    state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if not isinstance(state, Mapping):
        raise ValueError("Shared pretraining checkpoint has no model state_dict")
    state = {
        key.removeprefix("module."): tensor
        for key, tensor in state.items()
    }
    for prefix in (
        "global_tokens", "encoder.", "input_adapters.cxrs.",
        "input_adapters.xcs.", "output_adapters.cxrs.", "output_adapters.xcs."
    ):
        if not any(k == prefix or k.startswith(prefix) for k in state):
            raise ValueError(f"Shared pretraining checkpoint is missing {prefix} weights")
    return checkpoint, state


def load_shared_pretraining_weights(model, path, *, device="cpu", expected_model_name=None, regressor=False):
    """Load ALL expected pretrained tensors, fail before loading if even one differs.

    Args:
      regressor: True when the model wraps SpectralTF in ``backbone`` and has
        newly initialized ne, Ti and Te output heads.
    """
    checkpoint, source_state = read_verified_pretraining_checkpoint(
        path, device=device, expected_model_name=expected_model_name
    )
    target_state = model.state_dict()
    prefix = "backbone." if regressor else ""
    required = {
        key: tensor for key, tensor in target_state.items()
        if (
            key == prefix + "global_tokens"
            or key.startswith(prefix + "encoder.")
            or key.startswith(prefix + "input_adapters.")
            or (not regressor and key.startswith("output_adapters."))
        )
    }
    if not required:
        raise ValueError("Target model contains no SpectralTF pretrained backbone weights")
    # Spectral pretraining is multi-modal; each downstream loads only its active
    # input adapter and output decoder. Regression has no pretrain output decoder.
    missing = []
    mismatched = []
    weights = {}
    for target_key, target_tensor in required.items():
        source_key = target_key.removeprefix(prefix) if regressor else target_key
        tensor = source_state.get(source_key)
        if tensor is None:
            missing.append(source_key)
        elif tuple(tensor.shape) != tuple(target_tensor.shape):
            mismatched.append((source_key, tuple(tensor.shape), tuple(target_tensor.shape)))
        else:
            weights[target_key] = tensor

    if missing or mismatched:
        raise ValueError(
            "Shared pretraining checkpoint is NOT fully compatible. "
            f"Missing pretrained tensors: {missing[:8]}; "
            f"shape mismatches: {mismatched[:8]}. "
            "Check patch size / decoder / model shape and retrain the shared checkpoint."
        )
    result = model.load_state_dict(weights, strict=False)
    unexpected_missing = set(result.missing_keys) - (set(target_state) - set(required))
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Pretrained weights unexpectedly failed loading: "
            f"missing={unexpected_missing}, unexpected={result.unexpected_keys}"
        )
    print(
        f"Loaded {len(weights)}/{len(required)} compatible shared peak-aware "
        f"pretraining tensors from {path}; "
        f"{len(target_state) - len(required)} downstream-only tensors initialized normally."
    )
    return checkpoint
