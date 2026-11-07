#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume RGBD_revised_offline_v1 training from its saved last.pth.

Usage (run inside the RGBD_revised_offline_v1 project root):
  python -u train_rgbd_resume.py \
    --run-dir ./runs/formal_no_rgb_prior_s42 --target-epochs 100

The original trainer has no --resume flag. This companion uses its exact
model, dataset, loss, validation, and checkpoint helpers, restores model +
Adam optimizer state, and preserves existing history. It does NOT restore
unrecorded Python/NumPy/CUDA RNG or sampler state, so it is a continuation
from saved weights/optimizer, not a bit-identical uninterrupted run.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_rgbd import RGBDDataset
from models.RGBDRevised import RGBDRevisedNet
from train_rgbd_revised import (
    seed_everything, move_to_device, loss_fn, validate, checkpoint_save,
)

VARIANTS = {
    'baseline': (False, False, True),
    'pgdum': (True, False, True),
    'dapf': (False, True, True),
    'full': (True, True, True),
    'no_rgb_prior': (True, True, False),
}


def load_local_checkpoint(path):
    # Load only locally generated, trusted experiment checkpoints.
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Existing runs/formal_* folder')
    parser.add_argument('--target-epochs', type=int, default=100,
                        help='Final total epoch number (not additional epochs)')
    parser.add_argument('--expected-variant', default='no_rgb_prior', choices=list(VARIANTS),
                        help='Safety check: refuse to resume the wrong experiment')
    cli = parser.parse_args()

    run_dir = Path(cli.run_dir).expanduser().resolve()
    last = run_dir / 'last.pth'
    best_path = run_dir / 'best.pth'
    history_path = run_dir / 'history.json'
    for p in [last, best_path]:
        if not p.is_file():
            raise SystemExit(f'ERROR: Missing checkpoint: {p}')

    checkpoint = load_local_checkpoint(last)
    for key in ('epoch','model_state_dict','optimizer_state_dict','config'):
        if key not in checkpoint:
            raise SystemExit(f'ERROR: {last} missing {key}')
    config = dict(checkpoint['config'])
    variant = config.get('variant')
    if variant != cli.expected_variant:
        raise SystemExit(f'ERROR: Checkpoint variant={variant}, expected={cli.expected_variant}')
    first_epoch = int(checkpoint['epoch']) + 1
    if first_epoch > cli.target_epochs:
        print(f'Nothing to do: checkpoint already at epoch {checkpoint["epoch"]}.')
        return
    if first_epoch < 1:
        raise SystemExit('ERROR: Invalid checkpoint epoch')

    # Reuse original experiment hyperparameters and dataset split.
    args = argparse.Namespace(**config)
    args.save_dir = str(run_dir)
    args.epochs = cli.target_epochs
    original_dir = Path(str(config.get('save_dir', run_dir))).expanduser().resolve()
    if original_dir != run_dir:
        raise SystemExit(f'ERROR: Checkpoint save_dir is {original_dir}, requested {run_dir}')
    if args.epochs <= 0:
        raise SystemExit('ERROR: target epochs must be positive')

    best_ckpt = load_local_checkpoint(best_path)
    if best_ckpt.get('config', {}).get('variant') != variant:
        raise SystemExit('ERROR: best.pth variant differs from last.pth')
    if int(best_ckpt['epoch']) > int(checkpoint['epoch']):
        raise SystemExit('ERROR: best.pth epoch is newer than last.pth')
    best = float(best_ckpt['val_mae'])

    if history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))
        if not isinstance(history, list):
            raise SystemExit('ERROR: history.json must contain a list')
        if history and int(history[-1]['epoch']) > int(checkpoint['epoch']):
            raise SystemExit('ERROR: history.json extends beyond last.pth; inspect files before resuming')
    else:
        history = []
        print('WARNING: history.json absent; new history will only record resumed epochs')

    seed_everything(int(args.seed))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    base = Path(args.split_root)
    for split in ('train','val','test'):
        for sub in ('LR','GT','depth'):
            if not (base / split / sub).is_dir():
                raise SystemExit(f'ERROR: Missing split: {base / split / sub}')
    trainset = RGBDDataset(base/'train', side=args.size, augment=True,
                            corrupt_depth=True, gaussian_sigma=args.sigma,
                            dropout=args.dropout)
    valset = RGBDDataset(base/'val', side=args.size, augment=False, corrupt_depth=False)
    testset = RGBDDataset(base/'test', side=args.size, augment=False, corrupt_depth=False)
    sets = [set(x.stems) for x in (trainset,valset,testset)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise SystemExit('ERROR: Data leakage between splits')
    print('Train/Val/Test counts:', *[len(x) for x in (trainset,valset,testset)], flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=='cuda'),
                              generator=generator)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type=='cuda'))
    pgdum,dapf,rgb_prior = VARIANTS[variant]
    model = RGBDRevisedNet(use_pgdum=pgdum, use_dapf=dapf,
                           use_rgb_prior=rgb_prior).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Checkpoint was written by the original script after each validation epoch.
    # No scheduler is configured in the original training code; Adam state is kept.
    print(f'RESUME variant={variant}; checkpoint_epoch={checkpoint["epoch"]}; '
          f'next_epoch={first_epoch}; target_epochs={args.epochs}; '
          f'previous_best_val_mae={best:.9f}; device={device}', flush=True)
    print('NOTE: RNG/DataLoader states were not saved, so this is not bit-exact.', flush=True)
    (run_dir/'resume_configuration.json').write_text(
        json.dumps({'target_epochs':args.epochs,'source_checkpoint':str(last),
                    'source_epoch':checkpoint['epoch'],'original_config':config},
                   indent=2,ensure_ascii=False),encoding='utf-8')

    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        cumulative = 0.0
        n = 0
        for batch in train_loader:
            batch = move_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch['rgb'], batch['depth'], batch['valid'])
            loss, _ = loss_fn(out, batch, args)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            cumulative += loss.item()
            n += 1
        val_mae = validate(model, val_loader, device)
        row = {'epoch':epoch, 'train_loss':cumulative/max(n,1), 'val_mae':val_mae}
        print(json.dumps(row),flush=True)
        history.append(row)
        checkpoint_save(last, model, optimizer, epoch, val_mae, args)
        if val_mae < best:
            best = val_mae
            checkpoint_save(best_path, model, optimizer, epoch, val_mae, args)
        history_path.write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(f'COMPLETED {variant}: {args.epochs} epochs; best validation MAE={best:.9f}',flush=True)
    print('Final independent TEST is separate: evaluate_rgbd.py',flush=True)


if __name__ == '__main__':
    main()
