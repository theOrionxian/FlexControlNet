from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import deprecate
from diffusers.models.resnet import Downsample2D
from einops import rearrange

class DownsampleBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_padding: int = 1,
    ):
        super().__init__()

        self.downsamplers = nn.ModuleList(
            [
                Downsample2D(
                    in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                )
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)

        return hidden_states, (hidden_states,)

class DownsampleLoCon2D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        rank=8
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.name = name
        self.rank = rank
        self.kernel_size = kernel_size

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        std_norm = (rank*100)**0.25

        self.lora_a = nn.Parameter(torch.randn(self.out_channels, self.rank)/std_norm)
        self.lora_b = nn.Parameter(torch.randn(self.rank, channels*kernel_size*kernel_size)/std_norm)

        self.lora_c = nn.Parameter(torch.randn(self.out_channels, self.rank)/std_norm)
        self.lora_d = nn.Parameter(torch.randn(self.rank, channels*kernel_size*kernel_size)/std_norm)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = F.conv2d(
            hidden_states,
            rearrange((self.lora_a @ self.lora_b) * (self.lora_c @ self.lora_d), "o (i h w) -> o i h w", h=self.kernel_size, w=self.kernel_size),
            bias=self.bias,
            stride=2,
            padding=self.padding,
        )

        return hidden_states

class DownsampleLoConBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_padding: int = 1,
        rank=8
    ):
        super().__init__()

        self.downsamplers = nn.ModuleList(
            [
                DownsampleLoCon2D(
                    in_channels, out_channels=out_channels, padding=downsample_padding, name="op", rank=rank
                )
            ]
        )

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, temb: Optional[torch.Tensor] = None, *args, **kwargs
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)

        return hidden_states, (hidden_states,)