"""CPU shape/gradient test for patched Swin stage outputs.
If timm is missing, fake only timm utility functions to validate extraction.
"""
import sys
import types
from pathlib import Path
import torch
from torch import nn

try:
    import timm  # noqa
except ImportError:
    timm=types.ModuleType('timm'); timm.__path__=[]
    models=types.ModuleType('timm.models');models.__path__=[]
    layers=types.ModuleType('timm.models.layers')
    def to_2tuple(x):return x if isinstance(x,tuple) else (x,x)
    def trunc_normal_(tensor,std=.02):return nn.init.trunc_normal_(tensor,std=std)
    class DropPath(nn.Identity):
        def __init__(self,drop_prob=0):super().__init__()
    layers.DropPath=DropPath
    layers.to_2tuple=to_2tuple
    layers.trunc_normal_=trunc_normal_
    sys.modules.update({'timm':timm,'timm.models':models,'timm.models.layers':layers})

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.SwinTransformers import SwinTransformer
from models.RGBDRevised import RGBDRevisedNet

if __name__=='__main__':
    torch.manual_seed(123)
    backbone=SwinTransformer(img_size=64,embed_dim=8,depths=[1,1,1,1],
        num_heads=[1,1,2,2],window_size=4,drop_path_rate=0)
    f=backbone(torch.rand(2,3,64,64))
    expected=[(2,8,16,16),(2,16,8,8),(2,32,4,4),(2,64,2,2)]
    assert [tuple(x.shape) for x in f]==expected,[x.shape for x in f]
    print('Stage shapes:',[tuple(x.shape) for x in f])
    full=RGBDRevisedNet(channels=(8,16,32,64),rgb_backbone=backbone)
    out=full(torch.rand(2,3,64,64),torch.rand(2,1,64,64))
    loss=sum(x.square().mean() for x in out['sal'])
    loss.backward()
    last_stage_params=list(backbone.layers[-1].blocks[-1].parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in last_stage_params)
    print('Final Swin layer receives a nonzero gradient: YES')
    print('Swin+RGBDRevised CPU integration test: PASSED')
