"""Masked MSE reconstruction objective for SpectralTF."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedMSELoss(nn.Module):
    """MSE reconstruction loss with optional patch mask.

    Supports rectangular patch sizes, e.g. (3, 16) for CXRS
    and (5, 14) for XCS. The mask is defined at patch-token resolution
    with 1 indicating masked/reconstructed patches and 0 indicating visible patches.
    """

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))
        if isinstance(stride, (tuple, list)):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            self.stride = (int(stride), int(stride))
        self.scale_factor = (
            max(1, self.patch_size[0] // self.stride[0]),
            max(1, self.patch_size[1] // self.stride[1]),
        )
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p_h, p_w = self.scale_factor
        x = rearrange(
            imgs,
            "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)",
            nh=nh, nw=nw, p1=p_h, p2=p_w,
        )
        return x

    def unpatchify(self, x, nh, nw):
        p_h, p_w = self.scale_factor
        imgs = rearrange(
            x,
            "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)",
            nh=nh, nw=nw, p1=p_h, p2=p_w,
        )
        return imgs

    def forward(self, input, target, mask=None):
        H, W = input.shape[-2:]
        p_h, p_w = self.scale_factor
        nh, nw = H // p_h, W // p_w

        if target.shape[-2:] != input.shape[-2:]:
            target = target[..., :H, :W]

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)
            loss = loss * mask
            denom = mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
            loss = loss.flatten(start_dim=1).sum(dim=1) / denom
            loss = loss.nanmean()
        else:
            loss = loss.mean()

        return loss


