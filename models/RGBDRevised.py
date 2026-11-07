"""Research prototype for RGB-D SOD; NOT an experimentally validated method.

Differences from Wang et al., CVPR 2026 UMFNet:
  - one Swin RGB encoder plus a small dedicated depth CNN (not two Swin streams)
  - RGB edges / depth gradients condition depth uncertainty; no alignment objective
  - uncertainty can be supervised via corrupted-depth -> clean-depth Gaussian NLL
  - depth fusion is selective (shallow guidance plus two routed stages), not UAM/CGM everywhere
  - hard sample routing at evaluation skips unused branches (not sufficient by itself to prove speedup)

Requires torch; the real Swin encoder additionally requires timm.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def conv(in_ch, out_ch, stride=1, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(out_ch), nn.GELU()
    )


def resize_like(t, feat, mode='bilinear'):
    if t.shape[-2:] != feat.shape[-2:]:
        if mode == 'nearest':
            t = F.interpolate(t, size=feat.shape[-2:], mode=mode)
        else:
            t = F.interpolate(t, size=feat.shape[-2:], mode=mode, align_corners=False)
    return t


def sobel_magnitude(x):
    """Differentiable local gradient magnitude. Input Bx1xHxW."""
    assert x.ndim == 4 and x.shape[1] == 1
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3) / 8
    ky = kx.transpose(-2,-1)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square()+gy.square()+1e-8)
    denom = mag.detach().flatten(2).amax(dim=-1).view(-1,1,1,1).clamp_min(1e-4)
    return mag / denom


def structural_priors(rgb, depth, valid, target):
    # RGB channels are ImageNet-normalized; affine shifts vanish under gradient.
    lum = rgb[:,0:1] * .2989 + rgb[:,1:2] * .587 + rgb[:,2:3] * .114
    edge_rgb = sobel_magnitude(lum)
    # Do not take depth gradients across unknown pixels before masking.
    edge_depth = sobel_magnitude(depth * valid)
    er = resize_like(edge_rgb, target)
    ed = resize_like(edge_depth, target)
    vm = resize_like(valid, target, mode='nearest')
    return er, ed, torch.abs(er - ed), vm


class DepthEncoder(nn.Module):
    def __init__(self, channels=(128,256,512)):
        super().__init__()
        c1,c2,c3 = channels
        self.stem = conv(1,c1,stride=4,k=7)
        self.layer2 = conv(c1,c2,stride=2)
        self.layer3 = conv(c2,c3,stride=2)

    def forward(self, depth):
        a = self.stem(depth)
        b = self.layer2(a)
        c = self.layer3(b)
        return a,b,c


class PriorGuidedDUM(nn.Module):
    """Depth feature distribution conditioned on RGB/depth structure.

    `depth_mu` and `depth_logvar` parameterize a *scalar depth reconstruction*;
    their Gaussian NLL must be supervised using clean depth. The feature mean
    and depth confidence are used for fusion, not latent Gaussian alignment.
    """
    def __init__(self, channels, hidden=None, rgb_prior=True):
        super().__init__()
        h = hidden or max(16, channels//4)
        self.rgb_prior = rgb_prior
        self.prior = nn.Sequential(conv(4,h), conv(h,h))
        self.trunk = nn.Sequential(conv(channels*2+h,h), conv(h,h))
        self.feature_mean = nn.Conv2d(h,channels,1)
        self.depth_head = nn.Conv2d(h,2,1)
        self.conf_head = nn.Conv2d(h,1,1)

    def forward(self, rgb_f, depth_f, priors):
        er,ed,diff,valid = priors
        if not self.rgb_prior:
            er = torch.zeros_like(er)
            diff = torch.zeros_like(diff)
        p = self.prior(torch.cat((er,ed,diff,valid),dim=1))
        h = self.trunk(torch.cat((rgb_f,depth_f,p),dim=1))
        mu_f = self.feature_mean(h)
        depth_mu, depth_logvar = self.depth_head(h).chunk(2,dim=1)
        depth_logvar = depth_logvar.clamp(-6,4)
        # Learn confidence from local evidence and estimated aleatoric uncertainty.
        confidence = torch.sigmoid(self.conf_head(h) - .5*depth_logvar)
        confidence = valid * confidence
        refined_depth_f = depth_f + confidence * (mu_f - depth_f)
        return refined_depth_f, confidence, depth_mu, depth_logvar


def gaussian_depth_nll(mu, logvar, clean_depth, clean_valid):
    clean_depth = resize_like(clean_depth, mu)
    clean_valid = resize_like(clean_valid, mu,mode='nearest')
    sq = (mu-clean_depth).square()
    pixel_loss = 0.5 * (torch.exp(-logvar)*sq + logvar)
    return (pixel_loss * clean_valid).sum() / clean_valid.sum().clamp_min(1)


class DAPF(nn.Module):
    """Sample-level routing (RGB only / light / enhanced).

    Train: differentiable soft route, all 3 branches calculated.
    Eval: hard route using index_select/index_copy, unused branches skipped.
    The train/eval discrepancy requires empirical investigation.
    """
    def __init__(self, channels, hard_eval=True):
        super().__init__()
        h = max(16, channels//4)
        self.hard_eval = hard_eval
        self.light = nn.Conv2d(channels,channels,1,bias=False)
        self.enhanced = nn.Sequential(
            conv(channels*2+1,h),
            nn.Conv2d(h,h,3,padding=1,groups=h,bias=False),
            nn.GELU(),nn.Conv2d(h,channels,1,bias=False)
        )
        self.router = nn.Sequential(
            conv(channels*2+3,h),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),nn.Linear(h,3)
        )

    def forward(self, rgb_f, depth_f, confidence, edge_diff, depth_edge):
        valid = (confidence > 0).to(confidence.dtype)
        conf = confidence
        inp = torch.cat((rgb_f,depth_f,conf,edge_diff,valid),dim=1)
        logits = self.router(inp)
        probs = torch.softmax(logits,dim=1)
        if self.training or not self.hard_eval:
            rgb_only = rgb_f
            light = rgb_f + conf * self.light(depth_f)
            enhanced = rgb_f + conf * self.enhanced(torch.cat((rgb_f,depth_f,depth_edge),dim=1))
            out = (rgb_only * probs[:,0,None,None,None]
                   + light * probs[:,1,None,None,None]
                   + enhanced * probs[:,2,None,None,None])
        else:
            # Hard routing grouped by branch; no unused convolutional path runs.
            selected = probs.argmax(dim=1)
            out = rgb_f.clone()
            for branch in (1,2):
                ids = torch.nonzero(selected==branch,as_tuple=True)[0]
                if ids.numel()==0:
                    continue
                rv = rgb_f.index_select(0,ids)
                dv = depth_f.index_select(0,ids)
                cv = conf.index_select(0,ids)
                if branch == 1:
                    v = rv + cv*self.light(dv)
                else:
                    ed = depth_edge.index_select(0,ids)
                    v = rv + cv*self.enhanced(torch.cat((rv,dv,ed),dim=1))
                out.index_copy_(0,ids,v)
        # 0,1,3 are relative costs, NOT measured FLOPs.
        relative_route_cost = (probs * probs.new_tensor([0.,1.,3.])[None,:]).sum(dim=1).mean() / 3.
        return out, probs, relative_route_cost


class UpBlock(nn.Module):
    def __init__(self, upper,skip,out):
        super().__init__()
        self.body = nn.Sequential(conv(upper+skip,out),conv(out,out))
    def forward(self,up,skip):
        return self.body(torch.cat((resize_like(up,skip),skip),dim=1))


class Prediction(nn.Module):
    def __init__(self, channels):
        super().__init__()
        h=max(16,channels//2)
        self.sal=nn.Sequential(conv(channels,h),nn.Conv2d(h,1,1))
        self.bd=nn.Sequential(conv(channels,h),nn.Conv2d(h,1,1))
    def forward(self,x,size):
        return (F.interpolate(self.sal(x),size=size,mode='bilinear',align_corners=False),
                F.interpolate(self.bd(x),size=size,mode='bilinear',align_corners=False))


class RGBDRevisedNet(nn.Module):
    """Main revised model, using Swin for RGB and CNN for depth.

    For CPU unit tests pass a tiny `rgb_backbone` and four `channels`.
    """
    def __init__(self, channels=(128,256,512,1024), rgb_backbone=None,
                 use_pgdum=True, use_dapf=True, use_rgb_prior=True,
                 hard_eval=True):
        super().__init__()
        if rgb_backbone is None:
            from models.SwinTransformers import SwinTransformer
            rgb_backbone = SwinTransformer(embed_dim=channels[0],
                depths=[2,2,18,2],num_heads=[4,8,16,32])
        self.rgb_swin=rgb_backbone
        self.depth_encoder=DepthEncoder(channels[:3])
        c1,c2,c3,c4=channels
        self.use_pgdum=use_pgdum
        self.use_dapf=use_dapf
        self.pgdum1=PriorGuidedDUM(c1,rgb_prior=use_rgb_prior) if use_pgdum else None
        self.pgdum2=PriorGuidedDUM(c2,rgb_prior=use_rgb_prior) if use_pgdum else None
        self.shallow_depth_projection=nn.Conv2d(c1,c1,1)
        self.fuse2=DAPF(c2,hard_eval=hard_eval) if use_dapf else None
        self.fuse3=DAPF(c3,hard_eval=hard_eval) if use_dapf else None
        self.simple_fuse2=nn.Conv2d(c2,c2,1)
        self.simple_fuse3=nn.Conv2d(c3,c3,1)
        self.up3=UpBlock(c4,c3,c3)
        self.up2=UpBlock(c3,c2,c2)
        self.up1=UpBlock(c2,c1,c1)
        self.up0=nn.Sequential(conv(c1,c1//2),conv(c1//2,c1//2))
        self.predictions=nn.ModuleList([Prediction(c1//2),Prediction(c1),Prediction(c2),Prediction(c3)])

    def forward(self,rgb,depth,valid_mask=None):
        assert rgb.ndim==4 and rgb.shape[1]==3
        assert depth.ndim==4 and depth.shape[1]==1
        if valid_mask is None:
            valid_mask=(depth > 1e-6).float()
        depth = depth * valid_mask
        rf=self.rgb_swin(rgb)
        if len(rf)!=4:
            raise RuntimeError(f'RGB backbone must produce exactly 4 stages, got {len(rf)}')
        df=self.depth_encoder(depth)
        if any(r.shape != d.shape for r,d in zip(rf[:3],df)):
            raise RuntimeError('Depth/RGB stage shape mismatch: '+str([(r.shape,d.shape) for r,d in zip(rf[:3],df)]))
        clean_p=[structural_priors(rgb,depth,valid_mask,x) for x in rf[:3]]
        aux=[]
        refined=[]
        confidences=[]
        for i in range(2):
            if self.use_pgdum:
                dd,cc,mu,lv=getattr(self,f'pgdum{i+1}')(rf[i],df[i],clean_p[i])
                aux.append({'mu':mu,'logvar':lv})
            else:
                dd,cc=df[i],clean_p[i][3]
            refined.append(dd)
            confidences.append(cc)
        # Shallow structure guidance only at stage1 (no UAM/CGM chain).
        f1=rf[0]+confidences[0]*self.shallow_depth_projection(refined[0])
        # Stage2 and Stage3 route on difficulty; Stage4 stays RGB-semantic.
        f2_depth=refined[1]
        c2=confidences[1]
        c3=resize_like(c2,rf[2])
        ed2=clean_p[1][1]; ed3=clean_p[2][1]
        if self.fuse2 is not None:
            f2,p2,cost2=self.fuse2(rf[1],f2_depth,c2,clean_p[1][2],ed2)
        else:
            f2=rf[1]+c2*self.simple_fuse2(f2_depth)
            p2=cost2=None
        if self.fuse3 is not None:
            f3,p3,cost3=self.fuse3(rf[2],df[2],c3,clean_p[2][2],ed3)
        else:
            f3=rf[2]+c3*self.simple_fuse3(df[2])
            p3=cost3=None
        f4=rf[3]
        dec3=self.up3(f4,f3)
        dec2=self.up2(dec3,f2)
        dec1=self.up1(dec2,f1)
        dec0=F.interpolate(dec1,scale_factor=2,mode='bilinear',align_corners=False)
        dec0=self.up0(dec0)
        size=rgb.shape[-2:]
        pairs=[head(x,size) for head,x in zip(self.predictions,[dec0,dec1,dec2,dec3])]
        return {
            'sal':[p[0] for p in pairs],
            'bd':[p[1] for p in pairs],
            'uncertainty':aux,
            'routing_probs':[p for p in (p2,p3) if p is not None],
            'route_cost_proxy':torch.stack([p for p in (cost2,cost3) if p is not None]).mean() if self.use_dapf else rgb.new_zeros(())
        }
