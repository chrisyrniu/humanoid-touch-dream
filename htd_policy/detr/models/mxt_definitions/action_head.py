"""
Action head to adapt the output of the transformer to the action space. Adapted from HPT codebase.
"""

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from functools import partial
from typing import List
from detr.models.transformer import CrossAttention

LOSS = partial(F.smooth_l1_loss, beta=0.05, reduction='none')
INIT_CONST = 0.02


class ActionHead(nn.Module):
    """ Abstract class for policy head."""

    def __init__(self, **kwargs):
        super().__init__()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def save(self, path : str):
        torch.save(self.state_dict(), path)

    @property
    def device(self):
        return next(self.parameters()).device

    def compute_loss(self, x: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, is_pad: torch.Tensor, first_action: torch.Tensor=None, timestep_weights: torch.Tensor=None) -> torch.Tensor:
        self.target_action = label
        self.pred_action = self(x, first_action).view(self.target_action.shape)
        loss_list = LOSS(self.pred_action, self.target_action)
        action_horizon = self.target_action.shape[1]
        loss_list = loss_list * mask.unsqueeze(1)
        if timestep_weights is not None:
            loss_list = loss_list * timestep_weights.unsqueeze(-1)
        is_pad = is_pad[:, :action_horizon].unsqueeze(-1) if is_pad is not None else torch.zeros_like(self.target_action)
        # The final observation has no future dream targets. Its all-padded
        # supervision contributes a connected zero loss, rather than 0 / 0.
        mean_loss = torch.sum(loss_list * ~is_pad) / torch.sum(~is_pad).clamp(min=1)
        return mean_loss



class MLP(ActionHead):
    """Simple MLP based policy head"""

    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        dropout: bool = False,
        tanh_end: bool = False,
        ln: bool = True,
        **kwargs,
    ) -> None:
        """vanilla MLP head on the pooled feature"""
        super().__init__()
        self.input_dim = input_dim
        modules = [nn.Linear(input_dim, widths[0]), nn.SiLU()]

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if dropout:
                modules.append(nn.Dropout(p=0.1))
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def forward(self, x, first_action=None):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        y = self.net(x)
        if first_action is not None:
            y = y + first_action
        return y

class Conv1D(ActionHead):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        kernel_size: int = 3,
        n_groups: int = 8,
        action_horizon: int = 4,
        **kwargs,
    ) -> None:
        """
        Vanilla FFW head. Combined with learnable tokens imposed before the trunk.
        This one has a lower loss than MLP and should work better for long action chunks.
        """
        super().__init__()
        self.block_in = nn.Sequential(
            nn.Conv1d(input_dim, widths[0], kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, widths[0]),
            nn.Mish(),
        )
        output_dim = output_dim // action_horizon
        self.block_out = nn.Sequential(
            nn.Conv1d(widths[0], output_dim, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x, first_action=None):
        """
        Forward pass of the policy head module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        y = self.block_out(self.block_in(x)).view(len(x), -1) # flatten it
        if first_action is not None:
            y = y + first_action
        return y

class TransformerDecoderHead(ActionHead):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        crossattn_modality_dropout: float = 0.1,
        crossattn_heads: int = 8,
        crossattn_dim_head: int = 64,
        action_horizon: int = 4,
        **kwargs,
    ) -> None:
        """
        Transformer decoder similar to ACT or Detr head.
        This version uses cross attention and does not require retraining the trunk.
        """
        super().__init__()
        token_num = action_horizon
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, output_dim) * INIT_CONST
        )

        self.cross_attention = CrossAttention(
            output_dim,
            heads=crossattn_heads,
            dim_head=crossattn_dim_head,
            dropout=crossattn_modality_dropout,
        )
        embed_dim = crossattn_dim_head * crossattn_heads
        self.mlp = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, output_dim))
        self._register_query_pos(token_num, output_dim)

    def _register_query_pos(self, n, d):
        pos = np.arange(n)[:, np.newaxis]
        div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = np.zeros((n, d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.register_buffer('query_pos', torch.from_numpy(pe).float().unsqueeze(0))

    def forward(self, x: torch.Tensor, first_action=None) -> torch.Tensor:
        """
        Args:
        x: (B, token_len, input_dim)
        """
        context = self.mlp(x)
        context = context.reshape(context.shape[0], -1, context.shape[-1])
        queries = (self.tokens + self.query_pos).repeat(len(context), 1, 1)
        out = self.cross_attention(queries, context)
        # out[:, 1:, :] += out[:, 0, :].unsqueeze(1)
        # print("----RAW_OUT----", out[0])
        if first_action is not None:
            out = out + first_action.unsqueeze(1)
        # print("----OUT----", out[0])
        return out

############  Dream Head  ############

class TransformerDreamHead(ActionHead):
    """
    Dream head for predicting future touch representations (tactile latents or force values).
    Same architecture as TransformerDecoderHead — learnable query tokens cross-attend to trunk
    features to produce a sequence of future predictions.

    For tactile modalities: predicts the EMA-teacher-encoded latent of each future timestep.
    For force modalities:   predicts the z-score normalized force of each future timestep.
    """
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        crossattn_modality_dropout: float = 0.1,
        crossattn_heads: int = 8,
        crossattn_dim_head: int = 64,
        action_horizon: int = 4,
        use_projector: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        token_num = action_horizon
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, output_dim) * INIT_CONST
        )
        self.cross_attention = CrossAttention(
            output_dim,
            heads=crossattn_heads,
            dim_head=crossattn_dim_head,
            dropout=crossattn_modality_dropout,
        )
        embed_dim = crossattn_dim_head * crossattn_heads
        self.mlp = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, output_dim))
        self._register_query_pos(token_num, output_dim)
        if use_projector:
            proj_hidden = output_dim * 2
            self.projector = nn.Sequential(
                nn.Linear(output_dim, proj_hidden),
                nn.SiLU(),
                nn.Linear(proj_hidden, output_dim),
            )

    def _register_query_pos(self, n, d):
        pos = np.arange(n)[:, np.newaxis]
        div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = np.zeros((n, d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.register_buffer('query_pos', torch.from_numpy(pe).float().unsqueeze(0))

    def forward(self, x: torch.Tensor, first_action=None) -> torch.Tensor:
        """
        Args:
        x: (B, token_len, input_dim)
        """
        context = self.mlp(x)
        context = context.reshape(context.shape[0], -1, context.shape[-1])
        queries = (self.tokens + self.query_pos).repeat(len(context), 1, 1)
        out = self.cross_attention(queries, context)
        if first_action is not None:
            out = out + first_action.unsqueeze(1)
        return out
