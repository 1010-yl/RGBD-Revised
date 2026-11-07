"""Independent RGB-D test evaluation; never used for selecting checkpoints.

Corruption levels are proposal settings; use identical corruptions/seed for
all compared models. E_S matches the original script's max E-curve convention.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.RGBDRevised import RGBDRevisedNet
from data_rgbd import RGBDDataset


def evaluate(model,dl,device,noise,dropout,seed):
    try:
        from py_sod_metrics import Smeasure, Emeasure, WeightedFmeasure
    except ImportError as e:
        raise RuntimeError('pip install py-sod-metrics') from e
    sm,e,w=Smeasure(),Emeasure(),WeightedFmeasure()
    mae_total=0.
    n=0
    model.eval()
    with torch.inference_mode():
        for batch in dl:
            rgb,gt=batch['rgb'].to(device),batch['gt'].to(device)
            dep=batch['depth'].to(device).clone()
            valid=batch['valid'].to(device)
            if noise:
                # Per-sample deterministic noise controlled by one global torch generator.
                dep=(dep+torch.randn(dep.shape,device=device,generator=seed)*noise).clamp(0,1)*valid
            if dropout:
                keep=(torch.rand(dep.shape,device=device,generator=seed)>dropout).float()
                dep=dep*keep
            v=(dep>0).float()
            out=model(rgb,dep,v)
            pred=torch.sigmoid(sum(out['sal'])).cpu().numpy()
            gt_np=batch['gt'].cpu().numpy()
            for ii in range(pred.shape[0]):
                p=pred[ii,0]
                g=gt_np[ii,0]
                norm=(p-p.min())/(p.max()-p.min()+1e-8)
                pm=(norm*255).astype(np.uint8)
                gm=(g>0.5).astype(np.uint8)*255
                sm.step(pred=pm,gt=gm)
                e.step(pred=pm,gt=gm)
                w.step(pred=pm,gt=gm)
                mae_total+=np.abs(p-g).mean()
                n+=1
    e_res=e.get_results()['em']
    return {'S_alpha':float(sm.get_results()['sm']),
            'E_S':float(np.max(e_res['curve'])),
            'F_beta_w':float(w.get_results()['wfm']),
            'MAE':float(mae_total/n),'num_samples':n}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--test-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--size',type=int,default=384)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')
    args=p.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else ('cpu' if args.device=='auto' else args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ck['config']
    variants={'baseline':(False,False,True),'pgdum':(True,False,True),
              'dapf':(False,True,True),'full':(True,True,True),'no_rgb_prior':(True,True,False)}
    v=variants[cfg['variant']]
    net=RGBDRevisedNet(use_pgdum=v[0],use_dapf=v[1],use_rgb_prior=v[2]).to(device)
    net.load_state_dict(ck['model_state_dict'],strict=True)
    ds=RGBDDataset(args.test_root,side=args.size,augment=False,corrupt_depth=False)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    results=[]
    # All cases use the same clean samples and deterministic random state.
    for noise,drop in [(0,0),(.01,0),(.03,0),(.05,0),(0,.05),(0,.10),(0,.20),(0,1.)]:
        g=torch.Generator(device=device).manual_seed(args.seed)
        res=evaluate(net,dl,device,noise,drop,g)
        res.update({'gaussian_sigma':noise,'dropout_fraction':drop,'seed':args.seed,
                    'checkpoint':args.checkpoint,'variant':cfg['variant']})
        results.append(res)
        print(json.dumps(res,ensure_ascii=False))
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print('Saved',out)
