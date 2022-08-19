from mobileone_fpn import mobileone
import torch
from collections import OrderedDict
import torchvision
import torch.nn as nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from fcos_updated import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model import Backbone_FPN
def get_mobileone_s4_fpn_fcos(num_classes):
    backbone = mobileone(variant='s4', inference_mode=True)
    ckpt = '/home/mendeza/projects/mobilenetv3-fcos/mobileone_s4.pth.tar'
    checkpoint = torch.load(ckpt,map_location=torch.device('cpu'))
    backbone.load_state_dict(checkpoint)


    fpn = torchvision.ops.FeaturePyramidNetwork([64, 192, 448,896,2048],256)


    b_fpn = Backbone_FPN(backbone,fpn)
    b_fpn.out_channels = 256

    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
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
model = get_mobileone_s4_fpn_fcos(2)
model.eval()
print(model(torch.rand(1,3,224,224)))
# BackboneWithFPN(
#         backbone, 
#         [0,1,2,3,4], 
#         [64, 192, 448,896,2048], 
#         256, 
#         extra_blocks=LastLevelMaxPool(), 
#         norm_layer=None
#     )