# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import argparse
import functools
from typing import List
import einops
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer, build_transformer_decoder
from detr.models.mxt_definitions.tokenizer import ImageTokenizer, ProprioTokenizer, FingerTactileTokenizer, HandTactileTokenizer, HandPatchTactileEncoder
from detr.models.mxt_definitions.action_head import MLP, Conv1D, TransformerDecoderHead, TransformerDreamHead
from copy import deepcopy

import numpy as np

class SimpleNormalizer(nn.Module):
    def __init__(self, mean, std):
        super(SimpleNormalizer, self).__init__()
        std = torch.maximum(std, torch.tensor(1e-6))
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, x: torch.Tensor):
        return (x - self.mean) / self.std
    
    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def unnormalize(self, x: torch.Tensor):
        return x * self.std + self.mean
            
def dict_to_device(data, device):
    new_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            new_data[key] = dict_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            new_data[key] = value.to(device, non_blocking=True)
        else:
            new_data[key] = value
    return new_data

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


@functools.lru_cache(maxsize=128)
def get_sinusoid_encoding_table(n_position, d_hid):
    positions = np.arange(n_position)[:, np.newaxis]
    dim_pairs = np.arange(d_hid)[np.newaxis, :]
    angles = positions / np.power(10000, 2 * (dim_pairs // 2) / d_hid)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.FloatTensor(angles).unsqueeze(0) # 1, n_position, d_hidden


class CrossTransformer(nn.Module):
    ''' The cross transformer model, drawing from HPT and CrossFormer'''
    def __init__(
        self,
        embodiment_args_dict: dict = None,
        transformer_args: dict = None,
        norm_stats: dict = None,
        **kwargs,
        ):
        super().__init__()
        self.tokenizers = nn.ModuleDict()
        self.action_heads = nn.ModuleDict()
        self.dream_heads = nn.ModuleDict()
        self.ema_teachers = nn.ModuleDict()
        self.normalizers = nn.ModuleDict()
        self.dream_normalizers = nn.ModuleDict()
        self.query_len = dict()
        self.embodiment_args_dict = embodiment_args_dict
        self.transformer_args = transformer_args
        self.token_postprocessing = transformer_args["token_postprocessing"]
        for embodiment in [emb for emb in embodiment_args_dict.keys() if emb in norm_stats.keys()]:
            if isinstance(embodiment_args_dict[embodiment], dict):
                self.tokenizers[embodiment] = self._build_tokenizer(embodiment_args_dict[embodiment]["tokenizer"])
                self.action_heads[embodiment] = self._build_action_head(embodiment_args_dict[embodiment]["action_head"])
                self.normalizers[embodiment] = self._build_normalizers(embodiment_args_dict[embodiment], norm_stats[embodiment])
                if "dream_head" in embodiment_args_dict[embodiment]:
                    self.dream_heads[embodiment] = self._build_dream_head(embodiment_args_dict[embodiment]["dream_head"])
                    self.ema_teachers[embodiment] = self._build_ema_teachers(
                        embodiment_args_dict[embodiment], self.tokenizers[embodiment])
                    self.dream_normalizers[embodiment] = self._build_dream_normalizers(
                        embodiment_args_dict[embodiment]["dream_head"], norm_stats[embodiment])
        self._build_transformer_body(transformer_args)
        self.query_len = {modality: transformer_args["output_len"][modality] for modality in transformer_args["output_len"].keys()}
        total_query_len = int(np.sum(list(self.query_len.values())))
        self.register_buffer('query_pos_table',
                             get_sinusoid_encoding_table(total_query_len, self.transformer_body.d_model).squeeze(0))
        
    def _build_tokenizer(self, args):
        tokenizers = nn.ModuleDict()
        for modality in args.keys():
            if "image" in modality:
                tokenizers[modality] = ImageTokenizer(**args[modality])
                tokenizers[modality].init_cross_attn(args, modality)
            elif "proprio" in modality:
                tokenizers[modality] = ProprioTokenizer(**args[modality])
                tokenizers[modality].init_cross_attn(args, modality)
            elif "tactile" in modality:
                if "tactile_cap" in modality:
                    tokenizers[modality] = ProprioTokenizer(**args[modality])
                elif args[modality].get('input_dim', 0) == 1062:
                    tokenizers[modality] = HandTactileTokenizer(**args[modality])
                else:
                    tokenizers[modality] = FingerTactileTokenizer(**args[modality])
                tokenizers[modality].init_cross_attn(args, modality)
            # else:
            #     raise NotImplementedError
        total_params_tokenizers = sum([sum([p.numel() for p in tokenizers[modality].parameters() if p.requires_grad]) for modality in tokenizers.keys()])
        print(f"Total parameters in tokenizers: {total_params_tokenizers}")
        return tokenizers

    def _build_transformer_body(self, args, normalizer=None):
        self.trunk_mode = args["trunk_mode"]
        args_obj = argparse.Namespace()
        args_obj.__dict__.update(args)
        if self.trunk_mode == "encoder_decoder":
            self.transformer_body = build_transformer(args_obj)
            total_params_trunk = sum([p.numel() for p in self.transformer_body.parameters() if p.requires_grad])
            print(f"Total parameters in transformer body: {total_params_trunk}")
        elif self.trunk_mode == "decoder_only":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _build_action_head(self, args):
        action_heads = nn.ModuleDict()
        # normalizers = nn.ModuleDict()
        for modality in args.keys():
            if args[modality]["type"] == "mlp" :
                action_heads[modality] = MLP(**args[modality])
            elif args[modality]["type"] == "conv1d":
                action_heads[modality] = Conv1D(**args[modality])
            elif args[modality]["type"] == "transformer":
                action_heads[modality] = TransformerDecoderHead(**args[modality])
            else:
                raise NotImplementedError

        total_params_action_heads = sum([sum([p.numel() for p in action_heads[modality].parameters() if p.requires_grad]) for modality in action_heads.keys()])
        print(f"Total parameters in action heads: {total_params_action_heads}")
        return action_heads


    def _build_dream_head(self, args):
        dream_heads = nn.ModuleDict()
        for modality in args.keys():
            if args[modality].get("type") == "transformer":
                dream_heads[modality] = TransformerDreamHead(**args[modality])
            else:
                raise NotImplementedError(f"Unsupported dream head type: {args[modality].get('type')}")
        total_params = sum(p.numel() for m in dream_heads.values() for p in m.parameters() if p.requires_grad)
        print(f"Total parameters in dream heads: {total_params}")
        return dream_heads

    def _build_ema_teachers(self, embd_args, tokenizers):
        """Build EMA teacher copies of tactile tokenizer encoders (self.net only)."""
        ema_teachers = nn.ModuleDict()
        dream_keys = embd_args.get("dream_head", {}).keys()
        for key in dream_keys:
            if 'tactile' not in key or 'tactile_cap' in key:
                continue
            # find matching tokenizer (dream key matches tokenizer key for tactile)
            if key in tokenizers:
                teacher_net = deepcopy(tokenizers[key].net)
                for p in teacher_net.parameters():
                    p.requires_grad = False
                ema_teachers[key] = teacher_net
        if ema_teachers:
            print(f"Built EMA teachers for: {list(ema_teachers.keys())}")
        return ema_teachers

    @torch.no_grad()
    def update_ema(self):
        """Update EMA teacher parameters from student tokenizer nets."""
        for emb in self.ema_teachers.keys():
            decay = self.embodiment_args_dict[emb].get('ema_decay', 0.999)
            for dream_key, teacher_net in self.ema_teachers[emb].items():
                student_net = self.tokenizers[emb][dream_key].net
                for t_param, s_param in zip(teacher_net.parameters(), student_net.parameters()):
                    t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)

    @torch.no_grad()
    def encode_dream_labels(self, dream_data, embodiment):
        """Encode normalized dream label sequences with EMA teachers (tactile) or return as-is (force/raw)."""
        dream_labels = {}
        for key, data in dream_data.items():
            dream_raw = self.embodiment_args_dict.get(embodiment, {}).get('dream_head', {}).get(key, {}).get('dream_raw', False)
            if 'tactile' in key and not dream_raw and embodiment in self.ema_teachers and key in self.ema_teachers[embodiment]:
                B, T, D = data.shape
                flat = data.reshape(B * T, D)
                latent = self.ema_teachers[embodiment][key](flat)
                dream_labels[key] = latent.reshape(B, T, -1)
            else:
                dream_labels[key] = data
        return dream_labels

    def _build_dream_normalizers(self, dream_head_args, stats_dict):
        normalizers = nn.ModuleDict()
        dream_stats = stats_dict.get('dream', {})
        for key in dream_head_args.keys():
            if 'tactile' in key:
                continue
            if key in dream_stats and isinstance(dream_stats[key], dict):
                stats = dream_stats[key]
                normalizers[key] = SimpleNormalizer(
                    torch.from_numpy(stats['mean']).to(torch.float32),
                    torch.from_numpy(stats['std']).to(torch.float32))
        return normalizers

    def _build_normalizers(self, embd_args, stats_dict):
        obs_keys = embd_args['tokenizer'].keys()
        action_keys = embd_args['action_head'].keys()
        normalizers = nn.ModuleDict()
        for key in obs_keys:
            if 'proprio' in key:
                stats = stats_dict['obs'][key]
                normalizers[key] = SimpleNormalizer(torch.from_numpy(stats['mean']).to(torch.float32), torch.from_numpy(stats['std']).to(torch.float32))
            # tactile is kept as raw [0,1]
            elif 'image' in key:
                mean = torch.tensor([0.485, 0.456, 0.406]).to(torch.float32)
                std = torch.tensor([0.229, 0.224, 0.225]).to(torch.float32)
                normalizers[key] = SimpleNormalizer(mean, std)
        for key in action_keys:
            stats = stats_dict['action'][key]
            normalizers[key] = SimpleNormalizer(torch.from_numpy(stats['mean']).to(torch.float32), torch.from_numpy(stats['std']).to(torch.float32))
        return normalizers
            
    def tokenize(self, x, embodiment:str, x_mask):
        """
        Tokenize the input data to shape [batch, num_tokens, hidden_dim] for each modality and concat them
        """
        out = []
        out_mask = []
        bs = x[list(x.keys())[0]].shape[0]
        for modality in self.tokenizers[embodiment].keys():
            if modality not in x.keys():
                dummy_tokens = torch.zeros(bs,
                                           self.tokenizers[embodiment][modality].token_num,
                                           self.tokenizers[embodiment][modality].hidden_dim).to(x[list(x.keys())[0]].device)
                dummy_mask = torch.zeros(bs, self.tokenizers[embodiment][modality].token_num).to(x[list(x.keys())[0]].device)
                out.append(dummy_tokens)
                out_mask.append(dummy_mask)
                continue
            
            if not torch.any(x_mask[modality]):
                input_data = None
                output_tokens, output_mask = self.tokenizers[embodiment][modality].compute_latent(input_data, x_mask[modality])
                out.append(output_tokens)
                out_mask.append(output_mask)
            else:
                if 'proprio' in modality or 'tactile' in modality:
                    input_data = (x[modality] * x_mask[modality]).unsqueeze(-2) # apply mask
                else:
                    input_data = x[modality]
                features = self.tokenizers[embodiment][modality](input_data)
                
                data_shape = features.shape
                data_horizon = data_shape[1]
                horizon = data_horizon
                
                pos_embd = get_sinusoid_encoding_table(horizon * int(np.prod(data_shape[2:-1])), data_shape[-1]).to(features.device)
                pos_embd = einops.repeat(pos_embd, 'one n d -> b one n d', b=data_shape[0])
                features = features + pos_embd.view(data_shape)
                output_tokens, output_mask = self.tokenizers[embodiment][modality].compute_latent(features, x_mask[modality])
                out.append(output_tokens)
                out_mask.append(output_mask)
        out_mask = torch.cat(out_mask, dim=1)
        return out, out_mask
    
    def forward_features(self, x:dict, embodiment:str, x_mask=None, return_raw_trunk=False):
        """
        Forward the input data through the transformer body
        """
        for modality in x.keys():
            image_bytes = 'image' in modality and x[modality].dtype == torch.uint8
            x[modality] = x[modality].to(torch.float32)
            if image_bytes:
                # A tensor divisor preserves the CPU FP32 normalization on CUDA.
                x[modality].div_(x[modality].new_tensor(255.0))
        x = self.preprocess_states(x, embodiment)
        feature_tokens, feature_mask = self.tokenize(x, embodiment, x_mask)
        feature_tokens = self.preprocess_tokens(feature_tokens, embodiment) # added pos_embd
        query_pos_embd = self.query_pos_table
        dummy_features_pos_embd = torch.zeros_like(feature_tokens)[0].to(feature_tokens.device)
        trunk_features = self.transformer_body(feature_tokens, feature_mask, query_pos_embd, dummy_features_pos_embd)[-1]
        output_features_dict = self.partition_trunk_tokens(trunk_features, embodiment)
        for modality in output_features_dict.keys():
            output_features_dict[modality] = self.postprocess_tokens(output_features_dict[modality])
        if return_raw_trunk:
            return output_features_dict, trunk_features
        return output_features_dict
    
    def compute_loss(self, batch):
        device = next(self.parameters()).device
        batch = dict_to_device(batch, device)
        self.train_mode = True
        loss_dict = dict()
        loss_dict['all_emb_loss'] = 0
        for emb in batch.keys():
            loss_dict[emb] = dict()
            loss_dict[emb]['loss'] = 0
            obs_emb, obs_mask_emb = batch[emb]['obs'], batch[emb]['obs_mask']
            has_dream = emb in self.dream_heads and 'dream' in batch[emb]
            features_dict_emb = self.forward_features(obs_emb, emb, obs_mask_emb) if not has_dream \
                else self.forward_features(obs_emb, emb, obs_mask_emb, return_raw_trunk=True)
            if has_dream:
                features_dict_emb, trunk_features_raw = features_dict_emb

            action_emb, action_mask_emb = batch[emb]['action'], batch[emb]['action_mask']
            is_pad_emb = batch[emb]['is_pad']

            chunk_ts_weights = {}
            for modality in action_emb.keys():
                head_cfg = self.embodiment_args_dict[emb]['action_head'].get(modality, {})
                if isinstance(head_cfg, dict):
                    nzw = head_cfg.get('nonzero_chunk_weight', None)
                    if nzw is not None and nzw > 1.0:
                        thresh = head_cfg.get('nonzero_chunk_threshold', 0.10)
                        active = (action_emb[modality].abs().sum(dim=-1) > thresh).float()
                        chunk_ts_weights[modality] = 1.0 + active * (nzw - 1.0)

            if emb in self.normalizers.keys():
                for modality in action_emb.keys():
                    action_emb[modality] = self.normalizers[emb][modality].normalize(action_emb[modality])
            
            for modality in [modality for modality in action_emb.keys() if modality in features_dict_emb.keys()]:
                loss_dict[emb][modality] = self.action_heads[emb][modality].compute_loss(features_dict_emb[modality], action_emb[modality], action_mask_emb[modality], is_pad_emb, first_action=None, timestep_weights=chunk_ts_weights.get(modality, None))
                w = self.embodiment_args_dict[emb]['action_head'][modality].get('loss_weight', 1.0)
                loss_dict[emb][modality] = loss_dict[emb][modality] * w
                loss_dict[emb]['loss'] += loss_dict[emb][modality]

            # dream head losses
            if has_dream:
                dream_emb = batch[emb]['dream']
                dream_mask_emb = batch[emb]['dream_mask']
                is_pad_dream = batch[emb]['is_pad_dream']
                if emb in self.dream_normalizers:
                    for key in dream_emb:
                        if key in self.dream_normalizers[emb]:
                            dream_emb[key] = self.dream_normalizers[emb][key].normalize(dream_emb[key])
                # encode tactile labels with EMA teacher, force labels stay as-is
                dream_labels = self.encode_dream_labels(dream_emb, emb)
                if not self.training:
                    for key, lab in dream_labels.items():
                        if 'tactile' in key and 'tactile_cap' not in key and lab.dim() == 3:
                            flat = lab[~is_pad_dream[:, :lab.shape[1]]]
                            # Sample std and pairwise cosine need at least two valid targets.
                            if flat.shape[0] < 2:
                                continue
                            flat_norm = F.normalize(flat, dim=-1)
                            sim_matrix = flat_norm @ flat_norm.T
                            n = sim_matrix.shape[0]
                            mask_diag = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
                            mean_sim = sim_matrix[mask_diag].mean()
                            loss_dict[emb][f'ema_cosine_sim_{key}'] = mean_sim.detach()
                            loss_dict[emb][f'ema_std_{key}'] = flat.std(dim=0).mean().detach()
                for modality in self.dream_heads[emb].keys():
                    if modality not in dream_labels:
                        continue
                    label = dream_labels[modality]
                    dream_raw = self.embodiment_args_dict[emb]['dream_head'][modality].get('dream_raw', False)
                    if 'tactile' in modality and 'tactile_cap' not in modality and not dream_raw:
                        pred = self.dream_heads[emb][modality](trunk_features_raw)
                        pred = pred.view(label.shape)
                        if hasattr(self.dream_heads[emb][modality], 'projector'):
                            pred = self.dream_heads[emb][modality].projector(pred)
                        T = label.shape[1]
                        pad = is_pad_dream[:, :T]
                        use_cosine = self.embodiment_args_dict[emb]['dream_head'][modality].get('dream_ema_cosine', False)
                        if use_cosine:
                            direction_loss = 1.0 - F.cosine_similarity(pred, label, dim=-1)
                            magnitude_loss = F.smooth_l1_loss(pred.norm(dim=-1), label.norm(dim=-1), reduction='none')
                            beta = self.embodiment_args_dict[emb]['dream_head'][modality].get('dream_mag_beta', 0.1)
                            per_step_loss = direction_loss + beta * magnitude_loss
                        else:
                            per_step_loss = F.smooth_l1_loss(pred, label, reduction='none').mean(dim=-1)
                        dream_loss = torch.sum(per_step_loss * ~pad) / torch.sum(~pad).clamp(min=1)
                    else:
                        mask = torch.ones(label.shape[0], label.shape[-1], device=label.device).bool()
                        dream_loss = self.dream_heads[emb][modality].compute_loss(
                            trunk_features_raw, label,
                            mask, is_pad_dream, first_action=None)
                    w = self.embodiment_args_dict[emb]['dream_head'][modality].get('loss_weight', 1.0)
                    loss_dict[emb][f'dream_{modality}'] = dream_loss * w
                    loss_dict[emb]['loss'] += loss_dict[emb][f'dream_{modality}']

            loss_dict['all_emb_loss'] += loss_dict[emb]['loss']
            
        return loss_dict
    
    def forward(self, obs, embodiment, obs_mask=None, act_mask=None, infer_dream=False):
        """
        Forward the input data through the transformer body and action head.
        If infer_dream=True, also run dream heads and return their predictions.
        """
        device = next(self.parameters()).device
        obs = dict_to_device(obs, device)
        obs_mask = dict_to_device(obs_mask, device)
        has_dream = infer_dream and embodiment in self.dream_heads
        if has_dream:
            trunk_features_dict, trunk_features_raw = self.forward_features(obs, embodiment, obs_mask, return_raw_trunk=True)
        else:
            trunk_features_dict = self.forward_features(obs, embodiment, obs_mask)
        actions_dict = dict()
        for modality in self.action_heads[embodiment].keys():
            action = self.action_heads[embodiment][modality](trunk_features_dict[modality])
            action = action.view(1, self.embodiment_args_dict[embodiment]['action_head'][modality]['action_horizon'], -1)
            actions_dict[modality] = self.normalizers[embodiment][modality].unnormalize(action) if embodiment in self.normalizers.keys() else action
        if has_dream:
            dream_dict = dict()
            for modality in self.dream_heads[embodiment].keys():
                pred = self.dream_heads[embodiment][modality](trunk_features_raw)
                horizon = self.embodiment_args_dict[embodiment]['dream_head'][modality]['action_horizon']
                pred = pred.view(1, horizon, -1)
                if 'tactile' in modality and 'tactile_cap' not in modality and hasattr(self.dream_heads[embodiment][modality], 'projector'):
                    pred = self.dream_heads[embodiment][modality].projector(pred)
                dream_dict[modality] = pred
                if 'tactile' not in modality and embodiment in self.dream_normalizers:
                    if modality in self.dream_normalizers[embodiment]:
                        dream_dict[modality] = self.dream_normalizers[embodiment][modality].unnormalize(dream_dict[modality])
            return actions_dict, dream_dict
        return actions_dict
            
    def preprocess_tokens(self, features:List[torch.Tensor], embodiment:str):
        """
        Preprocess the tokenized data for the transformer
        """
        tokens = torch.cat(features, axis=-2)
        pos_embs = self.get_pos_embd(tokens)
        tokens = tokens + pos_embs
        return tokens

    def postprocess_tokens(self, tokens):
        """
        Postprocess the tokenized data after the transformer
        """
        if self.token_postprocessing == "mean":
            return tokens.mean(dim=1)
        elif self.token_postprocessing == "cls":
            return tokens[:, 0, :]
        elif self.token_postprocessing == "attentive":
            raise NotImplementedError
        elif self.token_postprocessing == "none":
            return tokens
    
    def preprocess_states(self, data:dict, embodiment:str):
        """
        Preprocess the proprioceptive data (tactile is kept as raw [0,1])
        """
        if self.embodiment_args_dict['normalize_state']:
            for modality in data.keys():
                if 'proprio' in modality:
                    data[modality] = self.normalizers[embodiment][modality].normalize(data[modality])
        
        return data
    
    def get_pos_embd(self, tokens):
        """
        Get the positional embedding for the tokens
        """
        token_len = tokens.shape[1]
        pos_embd = get_sinusoid_encoding_table(token_len, tokens.shape[-1]).to(tokens.device)
        return pos_embd
    
    def partition_trunk_tokens(self, tokens, embodiment:str):
        """
        Partition the tokens for the action head
        """
        modality_tokens = dict()
        for modality in self.query_len:
            modality_tokens[modality] = tokens[:, :self.query_len[modality], :]
            tokens = tokens[:, self.query_len[modality]:, :]
        return modality_tokens


def _concat_camera_features(features):
    """Keep the existing token order when camera feature heights match."""
    if len({feature.shape[2] for feature in features}) > 1:
        features = [feature.flatten(2).unsqueeze(2) for feature in features]
    return torch.cat(features, dim=3)


class DETRVAE_Decoder(nn.Module):
    """Non-autoregressive decoder-only ACT model."""
    def __init__(self, backbones, transformer_decoder, state_dim, num_queries, camera_names, action_dim,
                 feature_loss=False, backbone_map=None, tactile_config=None, proprio_dim=None):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.cam_num = len(camera_names)
        self.transformer_decoder = transformer_decoder
        self.state_dim, self.action_dim = state_dim, action_dim
        self.backbone_map = backbone_map
        hidden_dim = transformer_decoder.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.tactile_config = tactile_config
        if tactile_config is not None:
            self.proprio_dim = proprio_dim
            self.tactile_dim = tactile_config['tactile_dim']
            self.num_tactile_tokens = tactile_config['num_hands']
            enc_out = tactile_config['encoder_output_dim']
            self.tactile_encoder = HandPatchTactileEncoder(output_dim=enc_out)
            self.tactile_proj = nn.Linear(enc_out, hidden_dim)
            self.tactile_pos_embed = nn.Embedding(self.num_tactile_tokens, hidden_dim)
            proj_input_dim = proprio_dim
        else:
            self.proprio_dim = state_dim
            self.num_tactile_tokens = 0
            proj_input_dim = state_dim

        self.proprio_head = nn.Linear(hidden_dim, self.proprio_dim)

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(proj_input_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(proj_input_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None
        # encoder extra parameters
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq
        self.additional_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for proprio and latent
        self.feature_loss = feature_loss
    
    def forward(self, qpos, image):
        if self.feature_loss:
            if isinstance(image, dict):
                # Training supplies current/future views; rollout supplies current only.
                image_future = {name: frames[:, 1] for name, frames in image.items()
                                if frames.ndim == 5}
                image = {name: frames[:, 0] if frames.ndim == 5 else frames
                         for name, frames in image.items()}
            else:
                image_future = image[:,len(self.camera_names):].clone()
                image = image[:,:len(self.camera_names)].clone()

        all_cam_features = []
        all_cam_pos = []

        if isinstance(image, dict):
            if self.feature_loss and self.training:
                all_cam_features_future = []
            for cam_id, cam_name in enumerate(self.camera_names):
                bb_id = self.backbone_map[cam_id] if self.backbone_map is not None else (
                    cam_id if len(self.backbones) > 1 else 0
                )
                features, pos = self.backbones[bb_id](image[cam_name])
                all_cam_features.append(self.input_proj(features[0]))
                all_cam_pos.append(pos[0])
                if self.feature_loss and self.training:
                    features_f, _ = self.backbones[bb_id](image_future[cam_name])
                    all_cam_features_future.append(self.input_proj(features_f[0]))
        elif self.backbone_map is not None:
            if self.feature_loss and self.training:
                all_cam_features_future = []
            for cam_id, _ in enumerate(self.camera_names):
                bb_id = self.backbone_map[cam_id]
                features, pos = self.backbones[bb_id](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                if self.feature_loss and self.training:
                    features_f, _ = self.backbones[bb_id](image_future[:, cam_id])
                    all_cam_features_future.append(self.input_proj(features_f[0]))
        elif len(self.backbones)>1:
            # Image observation features and position embeddings
            for cam_id, _ in enumerate(self.camera_names):
                features, pos = self.backbones[cam_id](image[:, cam_id])
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                # all_cam_vis.append(vis)
        else:
            if self.feature_loss and self.training:
                all_cam_features_future = []
                bs,_,_,h,w = image.shape
                image_total = torch.cat([image, image_future], axis=0) #cat along the batch dimension
                bs_t,k,c,h_t,w_t = image_total.shape
                features, pos = self.backbones[0](image_total.reshape([-1,c,image_total.shape[-2],image_total.shape[-1]]))
                project_feature = self.input_proj(features[0])
                project_feature = project_feature.reshape([bs_t, self.cam_num, project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:bs,i,:])
                    all_cam_pos.append(pos[0])
                    all_cam_features_future.append(project_feature[bs:,i,:])
                    # all_cam_vis.append(vis)
            else:
                bs,_,c,h,w = image.shape
                features, pos = self.backbones[0](image.reshape([-1,c,h,w]))
                project_feature = self.input_proj(features[0]) 
                project_feature = project_feature.reshape([bs, self.cam_num,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for i in range(self.cam_num):
                    all_cam_features.append(project_feature[:,i,:])
                    all_cam_pos.append(pos[0])
                    # all_cam_vis.append(vis)
        # proprioception and tactile features
        if self.tactile_config is not None:
            proprio = qpos[:, :self.proprio_dim]
            tactile_raw = qpos[:, self.proprio_dim:]
            bs = proprio.shape[0]
            tactile_tokens_list = []
            for i in range(self.num_tactile_tokens):
                t = tactile_raw[:, i * self.tactile_dim:(i + 1) * self.tactile_dim]
                t_feat = self.tactile_proj(self.tactile_encoder(t))
                tactile_tokens_list.append(t_feat)
            extra_tokens = torch.stack(tactile_tokens_list, dim=0)  # (num_hands, B, hidden)
            extra_pos = self.tactile_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (num_hands, B, hidden)
            proprio_input = self.input_proj_robot_state(proprio)
        else:
            extra_tokens = None
            extra_pos = None
            proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = _concat_camera_features(all_cam_features)
        pos = _concat_camera_features(all_cam_pos)
        hs = self.transformer_decoder(src, self.query_embed.weight, proprio_input=proprio_input,
                                      pos_embed=pos, additional_pos_embed=self.additional_pos_embed.weight,
                                      extra_tokens=extra_tokens, extra_pos_embed=extra_pos)
        hs_action = hs[:,-1*self.num_queries:,:].clone()
        hs_img = hs[:,1:-(self.num_queries + self.num_tactile_tokens),:].clone() if self.num_tactile_tokens > 0 else hs[:,1:-1*self.num_queries,:].clone()
        hs_proprio = hs[:,[0],:].clone()
        a_hat = self.action_head(hs_action)
        a_proprio = self.proprio_head(hs_proprio) #proprio head
        if self.feature_loss and self.training:
            # proprioception features
            src_future = _concat_camera_features(all_cam_features_future)
            src_future = src_future.flatten(2).permute(2, 0, 1).transpose(1, 0) # B, 12*26, 512
            hs_img = {'hs_img': hs_img, 'src_future': src_future}
        return a_hat, a_proprio, hs_img
    
 
class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, vq, vq_class, vq_dim, action_dim,
                 backbone_map=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.backbone_map = backbone_map
        if self.backbone_map is not None:
            if len(self.backbone_map) != len(self.camera_names):
                raise ValueError(
                    f"backbone_map has {len(self.backbone_map)} entries for "
                    f"{len(self.camera_names)} cameras"
                )
            if not self.backbone_map or min(self.backbone_map) < 0 or max(self.backbone_map) >= len(backbones):
                raise ValueError(
                    f"backbone_map {self.backbone_map} is invalid for {len(backbones)} backbones"
                )
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.state_dim, self.action_dim = state_dim, action_dim
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding

        print(f'Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}')
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent


    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        bs, _ = qpos.shape
        if self.encoder is None:
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # cvae encoder
            is_training = actions is not None # train or val
            ### Obtain latent z from action sequence
            if is_training:
                # project action sequence to embedding dim, and concat with a CLS token
                action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
                qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                cls_embed = self.cls_embed.weight # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
                encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
                # do not mask cls token
                cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
                # obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
                # query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0] # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    probs = torch.softmax(logits, dim=-1)
                    binaries = F.one_hot(torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1), self.vq_dim).view(-1, self.vq_class, self.vq_dim).float()
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    straigt_through = binaries_flat - probs_flat.detach() + probs_flat
                    latent_input = self.latent_out_proj(straigt_through)
                    mu = logvar = None
                else:
                    probs = binaries = None
                    mu = latent_info[:, :self.latent_dim]
                    logvar = latent_info[:, self.latent_dim:]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)

            else:
                mu = logvar = binaries = probs = None
                if self.vq:
                    latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                else:
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, vq_sample=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)
        # cvae decoder
        # t = time.time()
        if self.backbones is not None:
            if isinstance(image, dict):
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    backbone_id = self.backbone_map[cam_id] if self.backbone_map is not None else (
                        cam_id if len(self.backbones) > 1 else 0
                    )
                    camera_image = image[cam_name]
                    if camera_image.ndim == 5:
                        camera_image = camera_image[:, 0]
                    features, pos = self.backbones[backbone_id](camera_image)
                    all_cam_features.append(self.input_proj(features[0]))
                    all_cam_pos.append(pos[0])
            elif self.backbone_map is not None:
                all_cam_features = []
                all_cam_pos = []
                for cam_id, _ in enumerate(self.camera_names):
                    backbone_id = self.backbone_map[cam_id]
                    features, pos = self.backbones[backbone_id](image[:, cam_id])
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
            elif len(self.backbones)>1:
                # Image observation features and position embeddings
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[cam_id](image[:, cam_id])
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
            else:
                all_cam_features = []
                all_cam_pos = []
                bs,k,c,h,w = image.shape
                image = image[:, :len(self.camera_names)] # exclude future images
                features, pos = self.backbones[0](image.reshape([-1, c, h, w]))
                project_feature = self.input_proj(features[0])
                project_feature = project_feature.reshape([bs, -1,project_feature.shape[1],project_feature.shape[2],project_feature.shape[3]])
                for cam_id in range(project_feature.shape[1]):
                    all_cam_features.append(project_feature[:,cam_id,:])
                    all_cam_pos.append(pos[0])
            # print(f'backbone time: {time.time()-t}')
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            
            # fold camera dimension into width dimension
            src = _concat_camera_features(all_cam_features)
            pos = _concat_camera_features(all_cam_pos)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[-1]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone_map = getattr(args, 'backbone_map', None)
    if backbone_map is not None:
        num_backbones = max(backbone_map) + 1
        for _ in range(num_backbones):
            backbones.append(build_backbone(args))
    elif args.same_backbones:
        backbone = build_backbone(args)
        backbones = [backbone]
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
        
    if args.no_vae_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    if args.model_type=="ACT":
        transformer = build_transformer(args)
        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            vq=args.vq,
            vq_class=args.vq_class,
            vq_dim=args.vq_dim,
            action_dim=args.action_dim,
            backbone_map=backbone_map,
        )
    elif args.model_type=="HIT":
        transformer_decoder = build_transformer_decoder(args)

        model = DETRVAE_Decoder(
            backbones,
            transformer_decoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            action_dim=args.action_dim,
            feature_loss= args.feature_loss if hasattr(args, 'feature_loss') else False,
            backbone_map=backbone_map,
            tactile_config=getattr(args, 'tactile_config', None),
            proprio_dim=getattr(args, 'proprio_dim', None),
        )
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_cross(args):
    model = CrossTransformer(
        embodiment_args_dict=args.embodiment_args_dict,
        transformer_args=args.transformer_args,
        norm_stats=args.norm_stats,
    )
    
    return model
