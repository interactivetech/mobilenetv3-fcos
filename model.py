# from fcos_updated import FCOS
import torch
import torchvision
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
# from mobileone import mobileone
from mobileone_fpn import mobileone

from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor
import torch.nn as nn
from torch import nn, Tensor
from typing import Callable, Dict, List, Optional, Union
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

class Backbone_FPN(nn.Module):
    def __init__(self,backbone: nn.Module,fpn: FeaturePyramidNetwork):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x: Tensor)-> Dict[str, Tensor]:
        y = self.backbone(x)
        x = self.fpn(y)
        return x


def get_mv3_fcos_fpn(num_classes):
    trainable_backbone_layers = 6
    # trainable_backbone_layers=5
    pretrained_backbone = False
    reduce_tail = True
    norm_layer = None
    b_model = torchvision.models.mobilenet_v3_large(pretrained=True)
    is_trained=True
    norm_layer = torchvision.ops.misc.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d
    backbone = _mobilenet_extractor(b_model, 
                                fpn=True,
                                trainable_layers=6,
                                returned_layers=[1,2,3,4,5])
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,),(256,))
    anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=((1.0,),)* len(anchor_sizes) 
    )   
    model = FCOS(
    backbone,
    num_classes=num_classes,
    anchor_generator=anchor_generator,
    )
    return model
     
def get_mv3_fcos_no_fpn(num_classes):

     backbone = torchvision.models.mobilenet_v2(pretrained=True).features
     # FCOS needs to know the number of
     # output channels in a backbone. For mobilenet_v2, it's 1280
     # so we need to add it here
     backbone.out_channels = 1280
    
     # let's make the network generate 5 x 3 anchors per spatial
     # location, with 5 different sizes and 3 different aspect
     # ratios. We have a Tuple[Tuple[int]] because each feature
     # map could potentially have different sizes and
     # aspect ratios
     anchor_sizes = ((128,))
     anchor_generator = AnchorGenerator(
         sizes=anchor_sizes,
         aspect_ratios=((1.0,),)
     )
    
     # put the pieces together inside a FCOS model
     model = FCOS(
         backbone,
         num_classes=num_classes,
         anchor_generator=anchor_generator,
     )
     return model

def get_mobileone_s4_no_fpn(num_classes):
    from mobileone_fpn import mobileone
    backbone = mobileone(variant='s4', inference_mode=True)
    ckpt = '/home/projects/mobilenetv3-fcos/mobileone_s4.pth.tar'
    checkpoint = torch.load(ckpt,map_location=torch.device('cuda:0'))
    backbone.load_state_dict(checkpoint)
    backbone=torch.nn.Sequential(*list(backbone.children())[:-2])

    backbone.out_channels = 2048

    anchor_sizes = ((32,))
    anchor_generator = AnchorGenerator(
         sizes=anchor_sizes,
         aspect_ratios=((1.0,),)
     )
    
    # put the pieces together inside a FCOS model
    model = FCOS(
         backbone,
         num_classes=num_classes,
         anchor_generator=anchor_generator,
     )
    return model

def get_mobileone_s4_fpn_fcos(num_classes):
    backbone = mobileone(variant='s4', inference_mode=True)
    ckpt = '/home/projects/mobilenetv3-fcos/mobileone_s4.pth.tar'
    checkpoint = torch.load(ckpt)
    backbone.load_state_dict(checkpoint)


    fpn = FeaturePyramidNetwork([ 64,192, 448,896,2048],256)
    # fpn = FeaturePyramidNetwork([ 64,192, 448,896],256)


    b_fpn = Backbone_FPN(backbone,fpn)
    b_fpn.out_channels = 256

    # anchor_sizes = ( (16,), (32,), (64,), (128,))
    anchor_sizes = ( (16,), (32,), (64,), (128,),(256,))

    anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=((1.0,),)* len(anchor_sizes) 
    )   
    model = FCOS(
    b_fpn,
    num_classes=num_classes,
    anchor_generator=anchor_generator,
    )
    return model