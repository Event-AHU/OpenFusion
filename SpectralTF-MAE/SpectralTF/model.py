# Copyright (c) EPFL VILAB.
# All rights reserved.

# --------------------------------------------------------
# Based on timm, DeiT, DINO, MoCo-v3, BEiT, MAE-priv and MAE code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# https://github.com/facebookresearch/moco-v3
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/BUPT-PRIV/MAE-priv
# https://github.com/facebookresearch/mae
# --------------------------------------------------------

from collections import OrderedDict
import math
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from einops import rearrange, repeat
from torch import nn

from utils.registry import register_model

from .transformer import Block, trunc_normal_

__all__ = [
    'pretrain_SpectralTF_base',
    'pretrain_SpectralTF_large',
]


class SpectralTF(nn.Module):
    """SpectralTF: joint peak-aware masked spectral Transformer
    This module performs masking in its forward pass.
    Downstream regression reuses the encoder with full, unmasked XCS tokens.


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """
    def __init__(self,
                 input_adapters: Dict[str, nn.Module],
                 output_adapters: Optional[Dict[str, nn.Module]],
                 num_global_tokens: int = 1,
                 dim_tokens: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        if output_adapters is not None:
            for adapter in output_adapters.values():
                adapter.init(dim_tokens_enc=dim_tokens)
            self.output_adapters = nn.ModuleDict(output_adapters)
        else:
            self.output_adapters = None

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)
        
        # Transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.encoder = nn.Sequential(*[
            Block(dim=dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        
        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {'global_tokens'}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, 'no_weight_decay'):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f'input_adapters.{task}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip
                
        if self.output_adapters is not None:
            for task, adapter in self.output_adapters.items():
                if hasattr(adapter, 'no_weight_decay'):
                    to_skip = adapter.no_weight_decay()
                    to_skip = set([f'output_adapters.{task}.{name}' for name in to_skip])
                    no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def generate_input_info(self, input_task_tokens, image_size, input_task_image_size=None):
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        input_task_image_size = input_task_image_size or {}
        for domain, tensor in input_task_tokens.items():
            num_tokens = tensor.shape[1]
            task_image_size = input_task_image_size.get(domain, image_size)
            adapter = self.input_adapters[domain]
            if hasattr(adapter, 'P_H') and hasattr(adapter, 'P_W'):
                grid_size = (task_image_size[0] // adapter.P_H, task_image_size[1] // adapter.P_W)
            else:
                grid_size = (1, num_tokens)
            d = {
                'num_tokens': num_tokens,
                'has_2d_posemb': True,
                'start_idx': i,
                'end_idx': i + num_tokens,
                'image_size': task_image_size,
                'grid_size': grid_size,
            }
            i += num_tokens
            input_info['tasks'][domain] = d

        input_info['image_size'] = image_size
        input_info['num_task_tokens'] = i
        input_info['num_global_tokens'] = self.num_global_tokens

        return input_info

    def forward(self, 
                x: Union[Dict[str, torch.Tensor], torch.Tensor], 
                mask_inputs: bool = True,
                task_masks: Dict[str, torch.Tensor] = None,
                num_encoded_tokens: int = 128,
                fp32_output_adapters: List[str] = []):
        """
        Forward pass through input adapters, transformer encoder and output adapters.
        During pretraining, selectively retains explicitly unmasked peak-aware tokens.

        :param x: Non-empty dictionary mapping spectral domains to tensors.
        :param mask_inputs: True requires supplied peak-aware task_masks; False keeps all patches
        :param task_masks: Optional dictionary of task->mask pairs.
        :param num_encoded_tokens: Number of tokens retained by the peak-aware masks.
            Only used if mask_inputs is True.
        :param fp32_output_adapters: List of task identifiers to force output adapters to
            run with mixed precision turned off for stability reasons.
        """

        if not isinstance(x, dict) or not x:
            raise TypeError("SpectralTF expects a non-empty {domain: [B, C, H, W]} input dictionary")
        first = next(iter(x.values()))
        B, C, H, W = first.shape
        input_task_image_size = OrderedDict()

        for domain, tensor in x.items():
            if domain not in self.input_adapters:
                raise KeyError(f"Input adapter is not registered for spectral domain: {domain}")
            if tensor.ndim != 4:
                raise ValueError(f"Spectral input {domain} must be [B, C, H, W]")
            _, _, h, w = tensor.shape
            input_task_image_size[domain] = (h, w)

        input_task_tokens = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(
            input_task_tokens=input_task_tokens,
            image_size=(H, W),
            input_task_image_size=input_task_image_size,
        )

        # Use explicit peak-aware masks for pretraining; keep all patches otherwise.
        if mask_inputs:
            if num_encoded_tokens is None:
                raise ValueError("Masked joint pretraining requires num_encoded_tokens")
        else:
            num_encoded_tokens = sum([tensor.shape[1] for tensor in input_task_tokens.values()])

        # Unmasked inference preserves all patches in spatial order.
        if not mask_inputs:
            total_tokens = sum(tensor.shape[1] for tensor in input_task_tokens.values())
            task_masks = {
                domain: torch.zeros((B, tensor.shape[1]), device=tensor.device, dtype=torch.long)
                for domain, tensor in input_task_tokens.items()
            }
            ids_keep = torch.arange(total_tokens, device=first.device).expand(B, -1)
            ids_restore = ids_keep
        else:
            if task_masks is None:
                raise ValueError(
                    "Masked pretraining requires explicit peak-aware task_masks. "
                    "Random MAE masking has been removed."
                )
            mask_all = torch.cat(
                [task_masks[task].to(input_task_tokens[task].device) for task in input_task_tokens.keys()],
                dim=1
            )

            # task_masks: 0 = keep, 1 = masked
            # Random jitter breaks ties when ordering visible and masked tokens.
            ids_shuffle = torch.argsort(mask_all.float() + 1e-6 * torch.rand_like(mask_all.float()), dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            num_keep_per_sample = (mask_all == 0).sum(dim=1)

            if not torch.all(num_keep_per_sample == num_keep_per_sample[0]):
                raise ValueError(
                    f'All samples must keep the same number of tokens when using task_masks, '
                    f'got {num_keep_per_sample.tolist()}'
                )

            num_keep = int(num_keep_per_sample[0].item())
            total_tokens = sum(tensor.shape[1] for tensor in input_task_tokens.values())
            if not 0 < num_keep < total_tokens or num_keep != num_encoded_tokens:
                raise ValueError(
                    f"Pretraining needs non-trivial peak-aware masking: "
                    f"kept={num_keep}, expected={num_encoded_tokens}, total={total_tokens}"
                )
            ids_keep = ids_shuffle[:, :num_keep]

        input_tokens = torch.cat([task_tokens for task_tokens in input_task_tokens.values()], dim=1)

        input_tokens = torch.gather(input_tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, input_tokens.shape[2]))

        global_tokens = repeat(self.global_tokens, '() n d -> b n d', b=B)
        input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        encoder_tokens = self.encoder(input_tokens)

        if self.output_adapters is None:
            return encoder_tokens, task_masks

        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
                ids_keep=ids_keep,
                ids_restore=ids_restore,
            )
            for domain in self.output_adapters
            if domain not in fp32_output_adapters
        }
        # Force running selected output adapters in fp32 mode
        with torch.amp.autocast("cuda", enabled=False):
            for domain in fp32_output_adapters:
                if domain not in self.output_adapters:
                    continue
                preds[domain] = self.output_adapters[domain](
                    encoder_tokens=encoder_tokens.float(),
                    input_info=input_info,
                    ids_keep=ids_keep,
                    ids_restore=ids_restore,
                )
        
        return preds, task_masks


@register_model
def pretrain_SpectralTF_base(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = SpectralTF(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

@register_model
def pretrain_SpectralTF_large(
        input_adapters: Dict[str, nn.Module],
        output_adapters: Optional[Dict[str, nn.Module]],
        **kwargs):
    model = SpectralTF(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        dim_tokens=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


