"""CPU-only synthetic tests; do not imply accuracy or GPU performance."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.RGBDRevised import RGBDRevisedNet, PriorGuidedDUM, DAPF, gaussian_depth_nll

class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages=nn.ModuleList([
            nn.Conv2d(3,8,3,padding=1),
            nn.Conv2d(8,16,3,padding=1),
            nn.Conv2d(16,32,3,padding=1),
            nn.Conv2d(32,64,3,padding=1)])
    def forward(self,rgb):
        x=rgb
        ret=[]
        for i,conv in enumerate(self.stages):
            x=F.avg_pool2d(x,4 if i==0 else 2)
            x=conv(x)
            ret.append(x)
        return ret

def run(use_pgdum=True,use_dapf=True,use_rgb_prior=True):
    torch.manual_seed(42)
    net=RGBDRevisedNet(channels=(8,16,32,64),rgb_backbone=FakeBackbone(),
             use_pgdum=use_pgdum,use_dapf=use_dapf,use_rgb_prior=use_rgb_prior)
    rgb=torch.rand((2,3,64,64))
    depth=torch.rand((2,1,64,64))
    depth[:,:,10:18,12:22]=0
    gt=torch.rand((2,1,64,64))
    out=net(rgb,depth)
    assert len(out['sal'])==4
    assert len(out['bd'])==4
    for t in out['sal']+out['bd']:
        assert t.shape == (2,1,64,64),t.shape
        assert torch.isfinite(t).all()
    loss=sum(F.binary_cross_entropy_with_logits(t,gt) for t in out['sal'])
    for aux in out['uncertainty']:
        loss=loss + 0.01*gaussian_depth_nll(aux['mu'],aux['logvar'],depth,(depth>0).float())
    loss=loss+0.001*out['route_cost_proxy']
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    net.eval()
    with torch.no_grad():
        out=net(rgb,depth)
    assert all(t.shape == (2,1,64,64) for t in out['sal'])
    return float(loss.detach())

if __name__=='__main__':
    print('full:',run(True,True,True))
    print('no prior:',run(True,True,False))
    print('no PGDUM:',run(False,True,True))
    print('no DAPF:',run(True,False,True))
    print('baseline:',run(False,False,True))
    print('CPU synthetic shape/backprop/hard-route tests: PASSED')
