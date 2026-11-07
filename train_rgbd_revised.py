"""Training entry for the research prototype; NO training has been run yet.

Example:
  python train_rgbd_revised.py --split-root /root/autodl-tmp/NJU2000_split_revised \
      --save-dir ./runs/full_s42 --variant full --seed 42 --epochs 100

Never trains on the test split. Best checkpoint chosen by VAL MAE only.
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data_rgbd import RGBDDataset
from models.RGBDRevised import RGBDRevisedNet,gaussian_depth_nll


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def soft_iou_loss(logits,target):
    p=torch.sigmoid(logits)
    inter=(p*target).sum(dim=(2,3))
    union=(p+target-p*target).sum(dim=(2,3))
    return (1-(inter+1)/(union+1)).mean()


def loss_fn(out,batch,args):
    gt=batch['gt']; bd=batch['boundary']
    ls=sum(F.binary_cross_entropy_with_logits(x,gt)+soft_iou_loss(x,gt) for x in out['sal'])
    lb=sum(F.binary_cross_entropy_with_logits(x,bd) for x in out['bd'])
    if out['uncertainty']:
        lu=torch.stack([
            gaussian_depth_nll(p['mu'],p['logvar'],batch['depth_clean'],batch['valid_clean'])
            for p in out['uncertainty']]).mean()
    else:
        lu=ls.new_zeros(())
    loss=ls+args.boundary_weight*lb+args.uncertainty_weight*lu+args.route_cost_weight*out['route_cost_proxy']
    return loss, {'sal':float(ls.detach()),'boundary':float(lb.detach()),
                  'uncertainty':float(lu.detach()),'route_cost_proxy':float(out['route_cost_proxy'].detach())}


def move_to_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def checkpoint_save(path,model,optimizer,epoch,val_mae,args):
    tmp=Path(str(path)+'.tmp')
    torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),
       'optimizer_state_dict':optimizer.state_dict(), 'val_mae':val_mae,'config':vars(args)},tmp)
    os.replace(tmp,path)


def validate(model,loader,device):
    model.eval()
    n=0; total=0.
    with torch.inference_mode():
        for batch in loader:
            batch=move_to_device(batch,device)
            out=model(batch['rgb'],batch['depth'],batch['valid'])
            pred=torch.sigmoid(sum(out['sal']))
            err=(pred-batch['gt']).abs().flatten(1).mean(dim=1)
            total+=err.sum().item()
            n+=err.shape[0]
    return total/max(n,1)


def main(args):
    seed_everything(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else (
        'cpu' if args.device=='auto' else args.device))
    base=Path(args.split_root)
    for split in ('train','val','test'):
        for sub in ('LR','GT','depth'):
            if not (base/split/sub).is_dir():
                raise FileNotFoundError(f'Missing split directory: {base/split/sub}')
    save=Path(args.save_dir)
    save.mkdir(parents=True,exist_ok=True)
    (save/'configuration.json').write_text(json.dumps(vars(args),ensure_ascii=False,indent=2),encoding='utf-8')
    trainset=RGBDDataset(base/'train',side=args.size,augment=True,
        corrupt_depth=True,gaussian_sigma=args.sigma,dropout=args.dropout)
    valset=RGBDDataset(base/'val',side=args.size,augment=False,corrupt_depth=False)
    testset=RGBDDataset(base/'test',side=args.size,augment=False,corrupt_depth=False)
    splitsets=[set(x.stems) for x in (trainset,valset,testset)]
    assert not (splitsets[0]&splitsets[1] or splitsets[0]&splitsets[2] or splitsets[1]&splitsets[2]), 'DATA LEAKAGE'
    print('Train/Val/Test counts:',*[len(x) for x in (trainset,valset,testset)])
    # The final test set is intentionally not used during training.
    gen=torch.Generator().manual_seed(args.seed)
    train_loader=DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=(device.type=='cuda'),generator=gen)
    val_loader=DataLoader(valset,batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=(device.type=='cuda'))
    variants={
        'baseline':(False,False,True),
        'pgdum':(True,False,True),
        'dapf':(False,True,True),
        'full':(True,True,True),
        'no_rgb_prior':(True,True,False),
    }
    pg,df,rgb_prior=variants[args.variant]
    model=RGBDRevisedNet(use_pgdum=pg,use_dapf=df,use_rgb_prior=rgb_prior).to(device)
    if args.swin_pretrained:
        checkpoint=torch.load(args.swin_pretrained,map_location='cpu',weights_only=False)
        sd=checkpoint.get('model',checkpoint.get('state_dict',checkpoint))
        incompat=model.rgb_swin.load_state_dict(sd,strict=False)
        n_matched=len(model.rgb_swin.state_dict())-len(incompat.missing_keys)
        if n_matched<20:
            raise RuntimeError(f'Only {n_matched} Swin keys matched. Verify checkpoint format.')
        print('Swin matched keys:',n_matched)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=float('inf')
    logs=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        cumulative=0.
        n=0
        for batch in train_loader:
            batch=move_to_device(batch,device)
            optimizer.zero_grad(set_to_none=True)
            out=model(batch['rgb'],batch['depth'],batch['valid'])
            loss,details=loss_fn(out,batch,args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
            optimizer.step()
            cumulative+=loss.item()
            n+=1
        val_mae=validate(model,val_loader,device)
        history={'epoch':epoch,'train_loss':cumulative/max(n,1),'val_mae':val_mae}
        logs.append(history)
        print(json.dumps(history))
        checkpoint_save(save/'last.pth',model,optimizer,epoch,val_mae,args)
        if val_mae<best:
            best=val_mae
            checkpoint_save(save/'best.pth',model,optimizer,epoch,val_mae,args)
        (save/'history.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    print('Best validation MAE:',best)
    print('Final TEST evaluation is separate: use evaluate_rgbd.py with best.pth')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--split-root',required=True)
    p.add_argument('--save-dir',required=True)
    p.add_argument('--variant',choices=['baseline','pgdum','dapf','full','no_rgb_prior'],default='full')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--size',type=int,default=384)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sigma',type=float,default=.03)
    p.add_argument('--dropout',type=float,default=.08)
    p.add_argument('--uncertainty-weight',type=float,default=.1)
    p.add_argument('--boundary-weight',type=float,default=.25)
    p.add_argument('--route-cost-weight',type=float,default=.001)
    p.add_argument('--clip-grad',type=float,default=1.)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='auto')
    p.add_argument('--swin-pretrained',default='')
    main(p.parse_args())
