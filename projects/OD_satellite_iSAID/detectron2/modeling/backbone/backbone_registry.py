from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from .resnet import ConvNeXt
from detectron2.modeling.backbone.fpn import FPN
import torch.nn as nn
import torch.nn.functional as F
import torch

# For convenience copied this class from detectron2/backbone/fpn.py
class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


# @BACKBONE_REGISTRY.register()
# def convnext_base(cfg, input_shape):
#     # def convnext_base(pretrained=False, in_22k=False, **kwargs):
#     model = ConvNeXt(cfg, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
#     # Do required settings here for now, like all backbone specific
#     # configurations
#     pretrained = True
#     in_22k = False # for now lets verfiy 1k pretrained backbone
#     if pretrained:
#         url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
#         checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
#         model.load_state_dict(checkpoint["model"])
#     return model
#
# @BACKBONE_REGISTRY.register()
# def build_convnext_fpn_backbone(cfg, input_shape):
#     bottom_up = convnext_base(cfg, input_shape)
#     in_features = cfg.MODEL.FPN.IN_FEATURES
#     out_channels = cfg.MODEL.FPN.OUT_CHANNELS
#     backbone = FPN(
#         bottom_up=bottom_up,
#         in_features=in_features,
#         out_channels=out_channels,
#         norm=cfg.MODEL.FPN.NORM,
#         top_block=LastLevelMaxPool(),
#         fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
#     )
#     return backbone