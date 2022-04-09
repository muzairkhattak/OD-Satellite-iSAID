# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers.batch_norm import get_norm, FrozenBatchNorm2d
from detectron2.modeling.backbone import Backbone

from timm import create_model

def freeze_module(x):
    """
    """
    for p in x.parameters():
        p.requires_grad = False
    FrozenBatchNorm2d.convert_frozen_batchnorm(x)
    return x

class TIMM(Backbone):
    def __init__(self, base_name, out_levels, freeze_at=0, norm='FrozenBN'):
        super().__init__()
        out_indices = [x - 1 for x in out_levels]

        self.base = create_model(
            base_name, features_only=True,
            out_indices=out_indices, pretrained=True)

        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction']) \
                        for i, f in enumerate(self.base.feature_info)]
        self._out_features = ['res{}'.format(x) for x in out_levels]
        self._out_feature_channels = {
            'res{}'.format(l): feature_info[l - 1]['num_chs'] for l in out_levels}
        self._out_feature_strides = {
            'res{}'.format(l): feature_info[l - 1]['reduction'] for l in out_levels}
        self._size_divisibility = max(self._out_feature_strides.values())
        if 'resnet' in base_name:
            self.freeze(freeze_at)
        if norm == 'FrozenBN':
            self = FrozenBatchNorm2d.convert_frozen_batchnorm(self)

    def freeze(self, freeze_at=0):
        """
        """
        if freeze_at >= 1:
            print('Frezing', self.base.conv1)
            self.base.conv1 = freeze_module(self.base.conv1)
        if freeze_at >= 2:
            print('Frezing', self.base.layer1)
            self.base.layer1 = freeze_module(self.base.layer1)

    def forward(self, x):
        features = self.base(x)
        ret = {k: v for k, v in zip(self._out_features, features)}
        return ret

    @property
    def size_divisibility(self):
        return self._size_divisibility


@BACKBONE_REGISTRY.register()
def build_timm_backbone(cfg, input_shape):
    model = TIMM(
        cfg.MODEL.TIMM.BASE_NAME,
        cfg.MODEL.TIMM.OUT_LEVELS,
        freeze_at=cfg.MODEL.TIMM.FREEZE_AT,
        norm=cfg.MODEL.TIMM.NORM,
    )
    return model
