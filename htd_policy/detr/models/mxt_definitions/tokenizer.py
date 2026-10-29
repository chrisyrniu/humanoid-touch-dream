from typing import List, Tuple
from detr.models.transformer import CrossAttention
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from torch.utils.checkpoint import checkpoint as grad_checkpoint

class BaseTokenizer(nn.Module):
    """
    Base tokenizer for the cross-transformer. Adapted from HPT codebase.
    """
    def __init__(self):
        super().__init__()
        
    def init_cross_attn(self, embodiment_tokenizer_args, modality:str):
        self.token_num = embodiment_tokenizer_args[modality]['token_num']
        self.tokens = nn.Parameter(torch.randn(1, self.token_num, embodiment_tokenizer_args['hidden_dim']))
        self.hidden_dim = embodiment_tokenizer_args['hidden_dim']
        self.cross_attn = CrossAttention(embodiment_tokenizer_args['hidden_dim'],
                                         embodiment_tokenizer_args['nhead'],
                                         embodiment_tokenizer_args['dim_head'],
                                         embodiment_tokenizer_args['dropout'])
        # Sinusoidal positional encoding for query tokens to encourage specialization
        d = self.hidden_dim
        pos = np.arange(self.token_num)[:, np.newaxis]
        div = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
        pe = np.zeros((self.token_num, d))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d // 2])
        self.register_buffer('query_pos', torch.from_numpy(pe).float().unsqueeze(0))
        
    def _get_query_tokens(self, bs):
        return (self.tokens + self.query_pos).repeat(bs, 1, 1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
            
    def save(self, path):
        torch.save(self.state_dict(), path)
            
    @property
    def device(self):
        return next(self.parameters()).device
    
    def compute_latent(self, feat):
        # Initial reshape to adapt to token dimensions
        features = feat.reshape(feat.shape[0], -1, feat.shape[-1])
        # Replicating tokens for each item in the batch and computing cross-attention
        query_tokens = self._get_query_tokens(len(features))
        output_tokens = self.cross_attn(query_tokens, features)
        return output_tokens
        
class ImageTokenizer(BaseTokenizer):
    def __init__(
        self,
        output_dim: int = 10,
        weights: str = "DEFAULT",
        resnet_model: str = "resnet18",
        num_of_copy: int = 1,
        split_dual: bool = False,
        gradient_checkpoint: bool = False,
        **kwargs
    ):
        super().__init__()
        pretrained_model = getattr(torchvision.models, resnet_model)(weights=weights)
        
        self.num_of_copy = num_of_copy
        self.split_dual = split_dual
        self.gradient_checkpoint = gradient_checkpoint
        feat_dim = pretrained_model.fc.in_features
        self.net = nn.Sequential(*list(pretrained_model.children())[:-2])
        
        if num_of_copy > 1:
            self.net = nn.ModuleList(
                [nn.Sequential(*list(pretrained_model.children())[:-2]) for _ in range(num_of_copy)]
            )
            
        self.out_dim = output_dim
        self.proj = nn.Linear(feat_dim, output_dim)
        
    def forward(self, x):
        """
        Args:
            x: Image tensor with shape [B, T, N, 3, H, W] representing the batch size, 
            horizon, instance (e.g. num of views)
        Returns:
            Flatten tensor with shape [B, M, output_dim]     
        """
        B, *_, H, W = x.shape
        x = x.view(len(x), -1, 3, H, W)
        use_ckpt = self.gradient_checkpoint and self.training
        if self.split_dual:
            W_half = W // 2
            x = x.reshape(-1, 3, H, W)
            x_left = x[:, :, :, :W_half]
            x_right = x[:, :, :, W_half:]
            if use_ckpt:
                feat_left = grad_checkpoint(self.net, x_left, use_reentrant=False)
                feat_right = grad_checkpoint(self.net, x_right, use_reentrant=False)
            else:
                feat_left = self.net(x_left)
                feat_right = self.net(x_right)
            feat = torch.cat([feat_left, feat_right], dim=-1)
        elif self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                if use_ckpt:
                    out.append(grad_checkpoint(net, input, use_reentrant=False))
                else:
                    out.append(net(input))
            feat = torch.stack(out, dim=1)
        else:
            x = x.view(-1, 3, H, W)
            if use_ckpt:
                feat = grad_checkpoint(self.net, x, use_reentrant=False)
            else:
                feat = self.net(x)
        feat = feat.flatten(start_dim=-2).transpose(1, 2)
        feat = self.proj(feat)
        feat = feat.reshape(B, -1, feat.shape[-1]).contiguous()
        return feat
    
    def compute_latent(self, feat, mask):
        # TODO: do we need to handle the case where mask is not the same for each sample in the batch?
        # for images, mask should be a single boolean (bs,)
        # Initial reshape to adapt to token dimensions
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            features = feat.reshape(bs, -1, feat.shape[-1])  # (bs, seq, out_dim)
            # Replicating tokens for each item in the batch and computing cross-attention
            query_tokens = self._get_query_tokens(bs)  # (bs, num_tokens, d_model)
            output_tokens = self.cross_attn(query_tokens, features)  # (bs, num_tokens, d_model)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask


class ProprioTokenizer(BaseTokenizer):
    def __init__(
        self,
        input_dim: int = 10,
        output_dim: int = 10,
        widths: List[int] = [512],
        tanh_end: bool = False,
        ln: bool = True,
        num_of_copy: int = 1,
        **kwargs
    ):
        super().__init__()
        modules = [nn.Linear(input_dim, widths[0])]
        if ln:
            modules.append(nn.LayerNorm(widths[0]))
        modules.append(nn.SiLU())

        for i in range(len(widths) - 1):
            modules.extend([nn.Linear(widths[i], widths[i + 1])])
            if ln:
                modules.append(nn.LayerNorm(widths[i + 1]))
            modules.append(nn.SiLU())

        modules.append(nn.Linear(widths[-1], output_dim))
        if tanh_end:
            modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self.num_of_copy = num_of_copy
        if self.num_of_copy > 1:
            self.net = nn.ModuleList([nn.Sequential(*modules) for _ in range(num_of_copy)])
        
    def forward(self, x):
        if self.num_of_copy > 1:
            out = []
            iter_num = min(self.num_of_copy, x.shape[1])
            for idx in range(iter_num):
                input = x[:, idx]
                net = self.net[idx]
                out.append(net(input))
            y = torch.stack(out, dim=1)
        else:
            y = self.net(x)
        return y
    
    def compute_latent(self, feat, mask):
        # Initial reshape to adapt to token dimensions
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            features = feat.reshape(bs, -1, feat.shape[-1])  # (bs, 1, d_model)
            # Replicating tokens for each item in the batch and computing cross-attention
            query_tokens = self._get_query_tokens(bs)  # (bs, num_tokens, d_model)
            output_tokens = self.cross_attn(query_tokens, features)  # (bs, num_tokens, d_model)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask


REGULAR_FINGER_PATCHES: List[Tuple[str, int, int, int]] = [
    ("tip",  0,   3, 3),
    ("top",  9,  12, 8),
    ("palm", 105, 10, 8),
]

THUMB_PATCHES: List[Tuple[str, int, int, int]] = [
    ("tip",  0,   3, 3),
    ("top",  9,  12, 8),
    ("mid",  105, 3, 3),
    ("palm", 114, 12, 8),
]

PALM_ONLY_PATCHES: List[Tuple[str, int, int, int]] = [
    ("palm", 0, 8, 14),
]


def _get_finger_patches(input_dim: int) -> List[Tuple[str, int, int, int]]:
    if input_dim == 185:
        return REGULAR_FINGER_PATCHES
    elif input_dim == 210:
        return THUMB_PATCHES
    elif input_dim == 112:
        return PALM_ONLY_PATCHES
    else:
        raise ValueError(f"Unknown finger tactile input_dim: {input_dim}")


class FingerPatchTactileEncoder(nn.Module):
    """CNN-per-patch encoder for a single finger's tactile array."""
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.patches = _get_finger_patches(input_dim)
        self.branches = nn.ModuleList()
        branch_out_channels = []
        for _, _, rows, cols in self.patches:
            if rows <= 3 and cols <= 3:
                branch = _make_small_cnn()
                out_c = 2
            elif (rows >= 8 and cols >= 14) or (rows >= 14 and cols >= 8):
                branch = _make_large_cnn()
                out_c = 3
            else:
                branch = _make_standard_cnn()
                out_c = 3
            self.branches.append(branch)
            branch_out_channels.append(out_c)
        total_flat = sum(c * POOL_SIZE * POOL_SIZE for c in branch_out_channels)
        self.pool = nn.AdaptiveAvgPool2d(POOL_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(total_flat, 2*total_flat),
            nn.SiLU(),
            nn.Linear(2*total_flat, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        B = flat.shape[0]

        parts = []
        for branch, (_, offset, rows, cols) in zip(self.branches, self.patches):
            size = rows * cols
            patch = flat[:, offset:offset + size]
            if rows == 8 and cols == 14:
                patch = patch.view(B, 1, 14, 8).transpose(-2, -1).contiguous()
            else:
                patch = patch.view(B, 1, rows, cols)
            feat = branch(patch)
            feat = self.pool(feat)
            parts.append(feat.flatten(1))

        out = self.mlp(torch.cat(parts, dim=1))
        return out.view(*leading, -1)


class FingerTactileTokenizer(BaseTokenizer):
    """Tactile tokenizer using per-patch CNNs for a single finger's tactile input."""
    def __init__(
        self,
        input_dim: int = 185,
        output_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        self.net = FingerPatchTactileEncoder(input_dim=input_dim, output_dim=output_dim)
        self._output_dim = output_dim

    def init_cross_attn(self, embodiment_tokenizer_args, modality: str):
        super().init_cross_attn(embodiment_tokenizer_args, modality)
        self.feat_to_token = nn.Linear(self._output_dim, self.hidden_dim)

    def forward(self, x):
        return self.net(x)

    def compute_latent(self, feat, mask):
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            feat = self.feat_to_token(feat)
            features = feat.reshape(bs, -1, feat.shape[-1])
            query_tokens = self._get_query_tokens(bs)
            output_tokens = self.cross_attn(query_tokens, features)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask

TACTILE_PATCHES: List[Tuple[str, int, int, int]] = [
    # (name, offset, rows, cols)
    ("fingerone_tip",       0,   3, 3),
    ("fingerone_top",       9,  12, 8),
    ("fingerone_palm",    105,  10, 8),
    ("fingertwo_tip",     185,   3, 3),
    ("fingertwo_top",     194,  12, 8),
    ("fingertwo_palm",    290,  10, 8),
    ("fingerthree_tip",   370,   3, 3),
    ("fingerthree_top",   379,  12, 8),
    ("fingerthree_palm",  475,  10, 8),
    ("fingerfour_tip",    555,   3, 3),
    ("fingerfour_top",    564,  12, 8),
    ("fingerfour_palm",   660,  10, 8),
    ("fingerfive_tip",    740,   3, 3),
    ("fingerfive_top",    749,  12, 8),
    ("fingerfive_mid",    845,   3, 3),
    ("fingerfive_palm",   854,  12, 8),
    ("palm",              950,   8, 14),  # transposed from (14,8) in raw data
]

POOL_SIZE = 4


def _make_small_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=2, stride=1),
        nn.SiLU(),
    )


def _make_standard_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2),
        nn.SiLU(),
        nn.Conv2d(2, 3, kernel_size=2, stride=1),
        nn.SiLU(),
    )


def _make_large_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=4, stride=2),
        nn.SiLU(),
        nn.Conv2d(2, 3, kernel_size=2, stride=1),
        nn.SiLU(),
    )


class HandPatchTactileEncoder(nn.Module):
    """CNN-per-patch encoder for a single hand's 1062-dim tactile array.
    
    Each of the 17 spatial patches is reshaped into a small 2-D feature map,
    processed by a dedicated CNN branch, adaptively pooled to POOL_SIZE x POOL_SIZE,
    flattened, concatenated, and fused via a global mixing MLP.
    """
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_out_channels = []
        for _, _, rows, cols in TACTILE_PATCHES:
            if rows <= 3 and cols <= 3:
                branch = _make_small_cnn()
                out_c = 2
            elif (rows >= 8 and cols >= 14) or (rows >= 14 and cols >= 8):
                branch = _make_large_cnn()
                out_c = 3
            else:
                branch = _make_standard_cnn()
                out_c = 3
            self.branches.append(branch)
            branch_out_channels.append(out_c)
        # 720 with POOL_SIZE=4, small=2ch, standard=3ch, large=3ch
        total_flat = sum(c * POOL_SIZE * POOL_SIZE for c in branch_out_channels)
        self.pool = nn.AdaptiveAvgPool2d(POOL_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(total_flat, total_flat),
            nn.SiLU(),
            nn.Linear(total_flat, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, 1062) — last dim is the flat tactile vector.
        Returns:
            (*, output_dim)
        """
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        B = flat.shape[0]

        parts = []
        for branch, (_, offset, rows, cols) in zip(self.branches, TACTILE_PATCHES):
            size = rows * cols
            patch = flat[:, offset:offset + size]
            if rows == 8 and cols == 14:
                patch = patch.view(B, 1, 14, 8).transpose(-2, -1).contiguous()
            else:
                patch = patch.view(B, 1, rows, cols)
            feat = branch(patch)
            feat = self.pool(feat)
            parts.append(feat.flatten(1))

        out = self.mlp(torch.cat(parts, dim=1))
        return out.view(*leading, -1)


class HandTactileTokenizer(BaseTokenizer):
    """Tactile tokenizer using per-patch CNNs for a full-hand 1062-dim input."""
    def __init__(
        self,
        input_dim: int = 1062,
        output_dim: int = 128,
        **kwargs
    ):
        super().__init__()
        self.net = HandPatchTactileEncoder(output_dim=output_dim)
        self._output_dim = output_dim

    def init_cross_attn(self, embodiment_tokenizer_args, modality: str):
        super().init_cross_attn(embodiment_tokenizer_args, modality)
        self.feat_to_token = nn.Linear(self._output_dim, self.hidden_dim)

    def forward(self, x):
        return self.net(x)

    def compute_latent(self, feat, mask):
        bs = mask.shape[0]
        if feat is None:
            output_tokens = torch.zeros((bs, self.token_num, self.hidden_dim)).to(self.device)
            trunk_mask = torch.zeros((bs, self.token_num)).to(self.device)
        else:
            feat = self.feat_to_token(feat)
            features = feat.reshape(bs, -1, feat.shape[-1])
            query_tokens = self._get_query_tokens(bs)
            output_tokens = self.cross_attn(query_tokens, features)
            trunk_mask = torch.ones((feat.shape[0], self.token_num)).to(self.device)
        return output_tokens, trunk_mask
