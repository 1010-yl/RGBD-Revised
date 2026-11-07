"""RGB-D dataset with synchronized geometry and a clean depth reference.

For NJU2000 grayscale depth visualizations: input values are normalized [0,1].
This is NOT a metric-depth/physical-sensor-noise calibration pipeline.
"""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps

EXTS={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def _lookup(path):
    d={}
    for p in Path(path).iterdir():
        if p.suffix.lower() in EXTS and p.is_file():
            if p.stem in d:
                raise ValueError(f'Duplicate stem: {p.stem}')
            d[p.stem]=p
    return d


def _depth_to_float(p):
    a=np.asarray(Image.open(p))
    if a.ndim==3:
        # Pseudo-colored maps are NOT metric depth; fail rather than silently corrupting.
        if not np.all(a[:,:,0]==a[:,:,1]) or not np.all(a[:,:,0]==a[:,:,2]):
            raise ValueError(f'Color depth image {p}; define calibrated depth conversion first')
        a=a[:,:,0]
    if a.dtype==np.uint8:
        scale=255.
    elif a.dtype==np.uint16:
        scale=65535.
    else:
        scale=max(float(np.nanmax(a)),1.)
    a=np.nan_to_num(a.astype(np.float32)/scale,nan=0.,posinf=0.,neginf=0.)
    return np.clip(a,0,1)


def _resize_depth_with_valid(depth,side):
    valid=(depth>0).astype(np.float32)
    d0=Image.fromarray(depth.astype(np.float32),mode='F').resize((side,side),Image.Resampling.BILINEAR)
    w0=Image.fromarray(valid,mode='F').resize((side,side),Image.Resampling.BILINEAR)
    w=np.asarray(w0,dtype=np.float32)
    out=np.asarray(d0,dtype=np.float32)/np.maximum(w,1e-5)
    # Preserve holes; do not transform missing pixels into reliable values.
    m=np.asarray(Image.fromarray(valid,mode='F').resize((side,side),Image.Resampling.NEAREST),dtype=np.float32)
    return np.clip(out,0,1)*m,m


class RGBDDataset(Dataset):
    def __init__(self, root, side=384, augment=False, corrupt_depth=False, gaussian_sigma=.03, dropout=.08):
        self.root=Path(root)
        self.side=side
        self.augment=augment
        self.corrupt_depth=corrupt_depth
        self.gaussian_sigma=gaussian_sigma
        self.dropout=dropout
        maps={k:_lookup(self.root/k) for k in ('LR','depth','GT')}
        self.maps=maps
        self.stems=sorted(set.intersection(*(set(v) for v in maps.values())))
        if not self.stems:
            raise RuntimeError(f'No matched RGB/Depth/GT samples in {root}')

    def __len__(self):
        return len(self.stems)

    def __getitem__(self,index):
        stem=self.stems[index]
        rgb=Image.open(self.maps['LR'][stem]).convert('RGB')
        gt=Image.open(self.maps['GT'][stem]).convert('L')
        dp=_depth_to_float(self.maps['depth'][stem])
        # Apply only synchronized horizontal flipping in this minimal version.
        flip=self.augment and random.random()<.5
        if flip:
            rgb=ImageOps.mirror(rgb)
            gt=ImageOps.mirror(gt)
            dp=np.fliplr(dp).copy()
        rgb=np.asarray(rgb.resize((self.side,self.side),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        gt=np.asarray(gt.resize((self.side,self.side),Image.Resampling.NEAREST),dtype=np.float32)/255.
        dp,valid=_resize_depth_with_valid(dp,self.side)
        rgb=(rgb-np.asarray([.485,.456,.406],dtype=np.float32))/np.asarray([.229,.224,.225],dtype=np.float32)
        rgb=torch.from_numpy(rgb.transpose(2,0,1).copy())
        gt=torch.from_numpy((gt>.5).astype(np.float32)[None,...].copy())
        depth_clean=torch.from_numpy(dp[None,...].copy())
        valid_clean=torch.from_numpy(valid[None,...].copy())
        corrupt=depth_clean.clone()
        if self.corrupt_depth:
            if self.gaussian_sigma:
                corrupt=(corrupt+torch.randn_like(corrupt)*self.gaussian_sigma).clamp(0,1)
            if self.dropout:
                keep=(torch.rand_like(corrupt)>self.dropout).float()
                corrupt=corrupt*keep
            corrupt=corrupt*valid_clean
        valid_corrupt=(corrupt>0).float()
        # Approximate morphological contour for separate boundary supervision.
        eroded=-torch.nn.functional.max_pool2d(-gt.unsqueeze(0),3,1,1).squeeze(0)
        boundary=(gt-eroded).clamp(0,1)
        return {'rgb':rgb,'gt':gt,'boundary':boundary,'depth_clean':depth_clean,
                'depth':corrupt,'valid_clean':valid_clean,'valid':valid_corrupt,'stem':stem}
