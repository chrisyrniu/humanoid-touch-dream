# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cross as build_cross


def build_ACT_model(args):
    return build_vae(args)


def build_MXT_model(args):
    return build_cross(args)
