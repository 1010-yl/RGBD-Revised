"""Create non-destructive NJU2000 train/val/test splits with an auditable manifest.

IMPORTANT: Never overwrites an existing directory. By default symlinks files,
not copies; preserve the original dataset and optionally preserve an existing
final test partition via --existing-test-root.
"""
import argparse
import json
import os
import random
from pathlib import Path

EXTS={'.jpg','.png','.bmp','.tif','.tiff'}

def mapping(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f'Missing directory {folder}')
    result={}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in EXTS:
            if p.stem in result:
                raise ValueError(f'Duplicate stem {p.stem}: {p} and {result[p.stem]}')
            result[p.stem]=p.resolve()
    return result

def run(args):
    src=Path(args.source_root).expanduser().resolve()
    out=Path(args.output_root).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Output exists; refusing to modify it: {out}')
    maps={k:mapping(src/k) for k in ('LR','GT','depth')}
    stems=sorted(set.intersection(*(set(m) for m in maps.values())))
    if not stems:
        raise RuntimeError(f'No paired image/GT/depth filenames in {src}')
    missed={k:len(m)-len(stems) for k,m in maps.items()}
    rng=random.Random(args.seed)
    if args.existing_test_root:
        existing=Path(args.existing_test_root).expanduser().resolve()
        existing_maps={k:mapping(existing/k) for k in ('LR','GT','depth')}
        test_stems=sorted(set.intersection(*(set(m) for m in existing_maps.values())))
        missing=sorted(set(test_stems)-set(stems))
        if missing:
            raise RuntimeError(f'Existing test has stems not present in source: {missing[:10]}')
        test_set=set(test_stems)
        pool=[s for s in stems if s not in test_set]
        rng.shuffle(pool)
    else:
        pool=stems[:]
        rng.shuffle(pool)
        test_stems=pool[:args.test_count]
        pool=pool[args.test_count:]
    if args.val_count >= len(pool) or args.val_count <= 0 or len(test_stems)==0:
        raise ValueError('Invalid split sizes; need nonempty train/val/test')
    val_stems=pool[:args.val_count]
    train_stems=pool[args.val_count:]
    assert len(set(train_stems)&set(val_stems))==0
    assert len(set(train_stems)&set(test_stems))==0
    assert len(set(val_stems)&set(test_stems))==0
    manifest={'source_root':str(src),'output_root':str(out),
              'existing_test_root':str(args.existing_test_root) if args.existing_test_root else None,
              'seed':args.seed,'test_count':len(test_stems),'val_count':len(val_stems),
              'train_count':len(train_stems),'unpaired_file_counts':missed,
              'splits':{'train':sorted(train_stems),'val':sorted(val_stems),'test':sorted(test_stems)}}
    out.mkdir(parents=True,exist_ok=False)
    for split,stems_part in manifest['splits'].items():
        for kind in ('LR','GT','depth'):
            dest=out/split/kind
            dest.mkdir(parents=True,exist_ok=True)
            for stem in stems_part:
                f=maps[kind][stem]
                dst=dest/f.name
                if args.copy:
                    import shutil
                    shutil.copy2(f,dst)
                else:
                    dst.symlink_to(f)
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Created {out}\ntrain={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")
    print('No original files were changed.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source-root',required=True,help='Original nju2000 with LR, GT, depth dirs')
    p.add_argument('--output-root',required=True,help='New output directory, must not already exist')
    p.add_argument('--existing-test-root',default='',help='Optional ORIGINAL NJU2000_split/test to freeze the existing test set')
    p.add_argument('--test-count',type=int,default=500)
    p.add_argument('--val-count',type=int,default=150)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--copy',action='store_true',help='Copy files instead of symlinks, uses extra disk')
    run(p.parse_args())
