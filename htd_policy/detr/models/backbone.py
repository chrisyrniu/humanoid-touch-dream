# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import List

import sys, os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, 
                 return_interm_layers: bool,
                 name: str):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        self.name = name
        if name.startswith('resnet'):
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        if name.startswith('resnet'):
            print(f'=================Using resnet {name}=================')
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
            num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, name)
    
    
class DINOv2BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.body.eval()
        self.num_channels = 384
        self.name = 'dino_v2'
    
    def forward(self, tensor):
        if self.training:
            with torch.no_grad():
                _, _, h, w = tensor.shape
                if w % 14 != 0 or h % 14 != 0:
                    new_w = w // 14 * 14
                    new_h = h // 14 * 14
                    # tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    tensor = tensor[:, :, :new_h, :new_w]
                else:
                    new_w = w
                    new_h = h
                xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
                od = OrderedDict()
                od["0"] = xs.reshape(xs.shape[0], new_w // 14, new_h // 14, 384).permute(0, 3, 2, 1)  # orig 22x16 tokens
                # return {"return": od, "visual": None}
                return od
        else:
            _, _, h, w = tensor.shape
            if w % 14 != 0 or h % 14 != 0:
                new_w = w // 14 * 14
                new_h = h // 14 * 14
                # tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                tensor = tensor[:, :, :new_h, :new_w]
            else:
                new_w = w
                new_h = h
            xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
            od = OrderedDict()
            od["0"] = xs.reshape(xs.shape[0], new_w // 14, new_h // 14, 384).permute(0, 3, 2, 1)  # orig 22x16 tokens
            # return {"return": od, "visual": None}
            return od

class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_channels = 768
        self.name = 'dummy'
        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, tensor):
        od = OrderedDict()
        od["0"] = self.pooling(tensor)
        return od
    
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list) # ["return"]
        # vis = self[0](tensor_list)["visual"]
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == 'dino_v2':
        backbone = DINOv2BackBone()
    elif args.backbone == 'dummy':
        backbone = DummyBackbone()
    else:
        assert args.backbone in ['resnet18', 'resnet34']
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
