# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse

import torch
from .models import build_ACT_model, build_MXT_model

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # ACT latent encoder
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class', required=False)
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim', required=False)
    parser.add_argument('--action_dim', action='store', type=int, required=False)
    parser.add_argument('--no_vae_encoder', action='store_true')
    parser.add_argument(
        '--no_encoder',
        dest='no_vae_encoder',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    # Policy transformer
    parser.add_argument('--model_type', action='store', type=str, default="HIT" ,help='model_type', required=False)
    parser.add_argument('--context_len', action='store', type=int, default=481, help='context_len', required=False)
    parser.add_argument('--use_pos_embd_image', action='store', type=int, default=0, required=False)
    parser.add_argument('--use_pos_embd_action', action='store', type=int, default=0, required=False)

    parser.add_argument('--self_attention', action='store', type=int, default=1, required=False)
    parser.add_argument('--same_backbones', action='store_true')

    # MXT optimizer groups
    parser.add_argument('--lr_tokenizer', type=float, required=False) # will be overridden
    parser.add_argument('--lr_action_head', type=float, required=False) # will be overridden
    parser.add_argument('--lr_dream_head', type=float, default=1e-6) # will be overridden
    parser.add_argument('--lr_trunk', type=float, required=False) # will be overridden
    return parser


def _normalize_act_encoder_flag(args_override):
    normalized = dict(args_override)
    legacy_value = normalized.pop('no_encoder', None)
    if legacy_value is not None:
        current_value = normalized.get('no_vae_encoder')
        if current_value is not None and bool(current_value) != bool(legacy_value):
            raise ValueError('no_encoder and no_vae_encoder disagree')
        normalized['no_vae_encoder'] = bool(legacy_value)
    return normalized


def build_ACT_model_and_optimizer(args_override):
    print("ARGS_OVERRIDE:", args_override)
    print("===========Building ACT model and optimizer===========")
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # Library builders use explicit configuration, independent of caller argv.
    args = parser.parse_args([])
    for k, v in _normalize_act_encoder_flag(args_override).items():
        print(f"Setting {k} to {v}")
        try:
            setattr(args, k, v)
        except:
            print(f"Error setting {k} to {v}")

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_MXT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    # This is a library builder: only its explicit configuration may override
    # defaults. Reading the caller's argv made inference and embedded training
    # depend on unrelated process flags.
    args = parser.parse_args([])
    
    for k, v in args_override.items():
        setattr(args, k, v)
        
    model = build_MXT_model(args)
    model.cuda()
    
    param_dicts = [
        # {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        # {
        #     "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        #     "lr": args.lr_backbone,
        # }
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "tokenizers" in n and "human" in n],
         "lr": args.lr_tokenizer},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "tokenizers" in n and "locoman" in n],
         "lr": args.lr_tokenizer},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "tokenizers" in n and "xarm7_leap" in n],
         "lr": args.lr_tokenizer},
         {"params": [p for n, p in model.named_parameters() if p.requires_grad and "tokenizers" in n and "g1_inspire_wb" in n],
         "lr": args.lr_tokenizer},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_heads" in n and "human" in n],
         "lr": args.lr_action_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_heads" in n and "locoman" in n],
         "lr": args.lr_action_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_heads" in n and "xarm7_leap" in n],
         "lr": args.lr_action_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "action_heads" in n and "g1_inspire_wb" in n],
         "lr": args.lr_action_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "dream_heads" in n],
         "lr": args.lr_dream_head},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "transformer_body" in n],
         "lr": args.lr_trunk},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and "tokenizers" not in n and "action_heads" not in n and "dream_heads" not in n and "transformer_body" not in n],
         "lr": args.lr}
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    return model, optimizer
