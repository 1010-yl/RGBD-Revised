#!/usr/bin/env bash
set -euo pipefail

cd /root/UMF/RGBD_revised_offline_v1

SPLIT=/root/UMF/NJU2000_split_revised
PRETRAIN=/root/UMF/extracted/UMFNet-main/swin_base_patch4_window12_384_22k.pth

for variant in baseline pgdum dapf full no_rgb_prior
do
    SAVE="./runs/formal_${variant}_s42"
    LOG="./logs/formal_${variant}_s42.log"

    if [ -e "$SAVE" ]; then
        echo "目录已存在，停止以避免覆盖：$SAVE"
        exit 1
    fi

    echo "========== START $variant =========="

    python -u train_rgbd_revised.py \
        --split-root "$SPLIT" \
        --save-dir "$SAVE" \
        --variant "$variant" \
        --seed 42 \
        --epochs 100 \
        --batch-size 1 \
        --workers 0 \
        --swin-pretrained "$PRETRAIN" \
        2>&1 | tee "$LOG"

    test -s "$SAVE/best.pth" || {
        echo "缺少最终 checkpoint：$variant"
        exit 1
    }

    echo "========== FINISHED $variant =========="
done

echo "ALL FIVE EXPERIMENTS FINISHED"
