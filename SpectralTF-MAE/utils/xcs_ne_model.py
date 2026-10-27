"""Model helpers for downstream XCS spectrum -> ne + Ti + Te regression."""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from SpectralTF.input_adapters import PatchedInputAdapter
from utils.model_builder import create_model
from utils.shared_checkpoint import load_shared_pretraining_weights

NE_DIM = 2
TI_DIM = 32
TE_DIM = 32
TARGET_DIM = NE_DIM + TI_DIM + TE_DIM


def build_xcs_ne_regression_head(output_dim: int, embed_dim: int) -> nn.Sequential:
    """Build the original one-hidden-layer regression head."""
    input_dim = int(embed_dim) * 2
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, output_dim),
    )


class XCSNeTiTePredictor(nn.Module):
    """Wrap a SpectralTF XCS backbone with independent ne/Ti/Te heads.

    The backbone representation is the concatenation of:
      1. mean pooled global token(s), shape [B, embed_dim]
      2. mean pooled local XCS patch tokens, shape [B, embed_dim]

    Each head therefore receives [B, 2 * embed_dim].  The returned tensor is ordered as:
      [ne1, ne2, Ti_0..Ti_31, Te_0..Te_31]  -> [B, 66]
    """

    def __init__(self, backbone: nn.Module, domain: str = "xcs", embed_dim: int = 768):
        super().__init__()
        self.backbone = backbone
        self.domain = domain
        self.embed_dim = int(embed_dim)
        self.ne_head = build_xcs_ne_regression_head(NE_DIM, self.embed_dim)
        self.ti_head = build_xcs_ne_regression_head(TI_DIM, self.embed_dim)
        self.te_head = build_xcs_ne_regression_head(TE_DIM, self.embed_dim)

    def no_weight_decay(self):
        skip = set()
        if hasattr(self.backbone, "no_weight_decay"):
            skip |= {f"backbone.{name}" for name in self.backbone.no_weight_decay()}
        return skip

    def extract_features(self, xcs: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            {self.domain: xcs},
            mask_inputs=False,
            task_masks=None,
        )
        encoder_tokens = out[0] if isinstance(out, tuple) else out

        num_global = int(getattr(self.backbone, "num_global_tokens", 1))
        if num_global > 0:
            local_tokens = encoder_tokens[:, :-num_global, :]
            global_tokens = encoder_tokens[:, -num_global:, :]
            pooled_global = global_tokens.mean(dim=1)
        else:
            local_tokens = encoder_tokens
            pooled_global = local_tokens.mean(dim=1)
        pooled_local = local_tokens.mean(dim=1)
        return torch.cat([pooled_global, pooled_local], dim=1)

    def forward(self, xcs: torch.Tensor, return_dict: bool = False):
        features = self.extract_features(xcs)
        outputs = {
            "ne": self.ne_head(features),
            "ti": self.ti_head(features),
            "te": self.te_head(features),
        }
        if return_dict:
            return outputs
        return torch.cat([outputs["ne"], outputs["ti"], outputs["te"]], dim=1)




def build_xcs_ne_model(args, domain_specs: Dict[str, Dict[str, Tuple[int, int]]]) -> XCSNeTiTePredictor:
    domain = "xcs"
    input_adapters = {
        domain: PatchedInputAdapter(
            num_channels=1,
            stride_level=1,
            patch_size_full=domain_specs[domain]["patch_size"],
            image_size=domain_specs[domain]["input_size"],
        )
    }
    backbone = create_model(
        args.model,
        input_adapters=input_adapters,
        output_adapters=None,
        num_global_tokens=args.num_global_tokens,
        drop_path_rate=args.drop_path,
    )
    embed_dim = int(backbone.global_tokens.shape[-1])
    return XCSNeTiTePredictor(backbone=backbone, domain=domain, embed_dim=embed_dim)


def load_pretrained_backbone_only(model: XCSNeTiTePredictor, ckpt_path: str, device="cpu", prefix="pretrained_ckpt", expected_model_name=None):
    """Load the SAME verified joint reconstruction checkpoint as spectral tasks.

    Reuse the full encoder and XCS patch projection.  The task-specific
    ne/Ti/Te prediction heads are intentionally initialized from scratch.
    """
    print(f"Loading shared {prefix} backbone: {ckpt_path}")
    return load_shared_pretraining_weights(
        model, ckpt_path, device=device, expected_model_name=expected_model_name,
        regressor=True
    )


def freeze_backbone_parts(model: XCSNeTiTePredictor, freeze_backbone=False, freeze_input_adapters=False, freeze_encoder=False):
    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        print("Frozen full backbone parameters; ne_head, ti_head, and te_head remain trainable.")
    else:
        if freeze_input_adapters and hasattr(model.backbone, "input_adapters"):
            for parameter in model.backbone.input_adapters.parameters():
                parameter.requires_grad = False
            print("Frozen input adapter parameters.")
        if freeze_encoder and hasattr(model.backbone, "encoder"):
            for parameter in model.backbone.encoder.parameters():
                parameter.requires_grad = False
            print("Frozen encoder parameters.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")
