import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont
from py_sod_metrics import WeightedFmeasure

from models.RGBDRevised import RGBDRevisedNet
from data_rgbd import RGBDDataset

ROOT = Path("/root/UMF/RGBD_revised_offline_v1")
TEST = Path("/root/UMF/NJU2000_split_revised/test")
OUT = ROOT / "runs/fig2_analysis_v2"

VARIANTS = {
    "baseline": (False, False, True),
    "pgdum": (True, False, True),
    "dapf": (False, True, True),
    "full": (True, True, True),
    "no_rgb_prior": (True, True, False),
}

LABELS = {
    "baseline": "Baseline",
    "pgdum": "PG-DUM",
    "dapf": "DAPF",
    "full": "Full",
    "no_rgb_prior": "w/o RGB Prior",
}

BILINEAR = Image.Resampling.BILINEAR
NEAREST = Image.Resampling.NEAREST

OUT.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)

dataset = RGBDDataset(
    TEST, side=384, augment=False, corrupt_depth=False
)

stems = list(dataset.stems)

if len(stems) != len(set(stems)):
    raise RuntimeError("Duplicate test sample IDs")

print("Test samples:", len(stems))

def index_files(folder):
    folder = TEST / folder
    result = {}
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in extensions:
            if p.stem in result:
                raise RuntimeError(f"Duplicate image stem: {p.stem}")
            result[p.stem] = p
    return result

FILES = {
    name: index_files(name)
    for name in ["LR", "depth", "GT"]
}

for stem in stems:
    for folder in FILES:
        if stem not in FILES[folder]:
            raise RuntimeError(f"Missing {folder}: {stem}")

def saliency_logits(output, size):
    if not isinstance(output, dict):
        raise TypeError("Unexpected model output")

    keys = [
        "saliency_logits",
        "sal_logits",
        "saliency_preds",
        "saliency",
        "sal",
        "sals",
    ]

    heads = None

    for key in keys:
        if key in output:
            heads = output[key]
            break

    if not isinstance(heads, (list, tuple)) or len(heads) != 4:
        raise RuntimeError(
            f"Cannot identify four saliency heads. "
            f"Actual keys: {list(output)}"
        )

    resized = [
        F.interpolate(
            head, size=size,
            mode="bilinear", align_corners=False
        )
        for head in heads
    ]
    return sum(resized)

def weighted_f(pred, gt):
    pred_u8 = np.clip(
        np.rint(pred * 255), 0, 255
    ).astype(np.uint8)

    gt_u8 = (
        (gt >= 0.5).astype(np.uint8) * 255
    )

    metric = WeightedFmeasure()
    metric.step(pred=pred_u8, gt=gt_u8)
    return float(metric.get_results()["wfm"])

records = {
    stem: {"stem": stem}
    for stem in stems
}

# --------------------------------------------------
# Run all five trained models on exactly the same
# 500 test samples. No test-time depth corruption.
# --------------------------------------------------

for variant, (use_pgdum, use_dapf, use_prior) in VARIANTS.items():

    print(f"\n========== {variant} ==========", flush=True)

    checkpoint_path = (
        ROOT / f"runs/formal_{variant}_s42/best.pth"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    model = RGBDRevisedNet(
        use_pgdum=use_pgdum,
        use_dapf=use_dapf,
        use_rgb_prior=use_prior,
    ).cuda().eval()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("config", {}).get("variant") != variant:
        raise RuntimeError(f"Checkpoint variant mismatch: {variant}")

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    pred_dir = OUT / "predictions" / variant
    pred_dir.mkdir(parents=True, exist_ok=True)

    for i, stem in enumerate(stems):

        sample = dataset[i]

        rgb = sample["rgb"].unsqueeze(0).cuda()
        depth = sample["depth"].unsqueeze(0).cuda()
        valid = sample["valid"].unsqueeze(0).cuda()

        # Evaluation-resolution prediction (384 x 384).
        logits = saliency_logits(
            model(rgb, depth, valid),
            rgb.shape[-2:]
        )

        prob = (
            logits.sigmoid()[0, 0]
            .detach().cpu().numpy()
            .astype(np.float32)
        )

        gt_eval = (
            sample["gt"].detach().cpu()
            .numpy().squeeze().astype(np.float32)
        )

        if prob.shape != gt_eval.shape:
            raise RuntimeError(
                f"Shape mismatch {stem}: "
                f"pred={prob.shape}, GT={gt_eval.shape}"
            )

        gt_eval = np.clip(gt_eval, 0, 1)

        mae = float(np.abs(prob - gt_eval).mean())
        fw = weighted_f(prob, gt_eval)

        records[stem][f"{variant}_MAE"] = mae
        records[stem][f"{variant}_Fw"] = fw

        # Restore prediction to native GT coordinates
        # BEFORE creating the visualization.
        with Image.open(FILES["GT"][stem]) as gt_file:
            native_w, native_h = gt_file.size

        native_logits = F.interpolate(
            logits,
            size=(native_h, native_w),
            mode="bilinear",
            align_corners=False,
        )

        native_prob = (
            native_logits.sigmoid()[0, 0]
            .detach().cpu().numpy()
        )

        pred_u8 = np.clip(
            np.rint(native_prob * 255), 0, 255
        ).astype(np.uint8)

        Image.fromarray(pred_u8, "L").save(
            pred_dir / f"{stem}.png"
        )

        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(stems)}", flush=True)

    del model
    torch.cuda.empty_cache()

# --------------------------------------------------
# Calculate per-sample gains.
# Positive MAE gain = improvement.
# Positive Fw gain = improvement.
# --------------------------------------------------

rows = []

for stem in stems:
    r = records[stem]

    r["full_vs_baseline_MAE_gain"] = (
        r["baseline_MAE"] - r["full_MAE"]
    )
    r["full_vs_baseline_Fw_gain"] = (
        r["full_Fw"] - r["baseline_Fw"]
    )

    r["pgdum_vs_full_MAE_gain"] = (
        r["full_MAE"] - r["pgdum_MAE"]
    )
    r["pgdum_vs_full_Fw_gain"] = (
        r["pgdum_Fw"] - r["full_Fw"]
    )

    rows.append(r)

fields = list(rows[0].keys())

with (OUT / "per_image_metrics.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("\n========== CLEAN METRIC MEANS ==========")

for variant in VARIANTS:
    mae = np.mean([r[f"{variant}_MAE"] for r in rows])
    fw = np.mean([r[f"{variant}_Fw"] for r in rows])

    print(f"{variant:15s} MAE={mae:.6f} Fw={fw:.6f}")

    # Compare with the earlier official evaluation CSV.
    reference = (
        ROOT / f"runs/formal_{variant}_s42/"
        "test_noise_metrics.csv"
    )

    if reference.is_file():
        with reference.open(
            newline="", encoding="utf-8-sig"
        ) as f:
            entries = list(csv.DictReader(f))

        clean = [
            x for x in entries
            if float(x["gaussian_sigma"]) == 0
            and float(x["dropout_fraction"]) == 0
        ]

        if len(clean) == 1:
            old_mae = float(clean[0]["MAE"])
            old_fw = float(clean[0]["F_beta_w"])

            print(
                f"  Reference: MAE={old_mae:.6f} "
                f"Fw={old_fw:.6f}"
            )

            if (
                abs(mae - old_mae) > 0.001
                or abs(fw - old_fw) > 0.001
            ):
                print(
                    "  WARNING: Metric protocol differs "
                    "from evaluate_rgbd.py. "
                    "Do not merge these values into "
                    "the paper without reconciliation."
                )

# --------------------------------------------------
# Rank by two agreeing metrics.
# Also export complete rankings for inspection.
# --------------------------------------------------

groups = {
    "full_improved": (
        "full_vs_baseline_MAE_gain",
        "full_vs_baseline_Fw_gain",
        1,
    ),
    "full_regressed": (
        "full_vs_baseline_MAE_gain",
        "full_vs_baseline_Fw_gain",
        -1,
    ),
    "pgdum_better_than_full": (
        "pgdum_vs_full_MAE_gain",
        "pgdum_vs_full_Fw_gain",
        1,
    ),
}

selected = {}

for group, (mae_key, fw_key, sign) in groups.items():

    ranked = sorted(
        rows,
        key=lambda r: sign * r[mae_key],
        reverse=True,
    )

    with (OUT / f"ranking_{group}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked)

    # Only select cases where MAE and Fw agree.
    eligible = [
        r for r in ranked
        if sign * r[mae_key] > 0
        and sign * r[fw_key] > 0
    ]

    selected[group] = eligible[:4]

    print(f"\n========== {group} ==========")
    print("Cases meeting both criteria:", len(eligible))

    for r in selected[group]:
        print(
            r["stem"],
            f"MAE gain={r[mae_key]:+.6f}",
            f"Fw gain={r[fw_key]:+.6f}",
        )

# --------------------------------------------------
# Create publication-style figures.
# One physical GT coordinate system per row.
# All columns share identical aspect-preserving
# scaling and padding.
# --------------------------------------------------

COLUMNS = [
    "RGB", "Depth", "GT",
    "baseline", "pgdum", "dapf",
    "full", "no_rgb_prior",
]

CELL = 204
GAP = 9
ROW_GAP = 12
HEADER = 56
ROW_LABEL = 27

try:
    FONT = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        17,
    )
    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        13,
    )
except OSError:
    FONT = ImageFont.load_default()
    FONT_SMALL = FONT

def display_tile(image, size, background):
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (size, size),
        method=BILINEAR,
    )

    tile = Image.new(
        "RGB", (size, size), background
    )

    tile.paste(
        fitted,
        ((size - fitted.width)//2,
         (size - fitted.height)//2),
    )
    return tile

def compose(group_name, items):
    if not items:
        print("No qualifying examples:", group_name)
        return

    width = GAP + len(COLUMNS)*(CELL+GAP)
    row_height = ROW_LABEL + CELL + ROW_GAP
    height = HEADER + len(items)*row_height + GAP

    canvas = Image.new(
        "RGB", (width, height), "#FFFFFF"
    )
    draw = ImageDraw.Draw(canvas)

    for j, col in enumerate(COLUMNS):
        label = LABELS.get(col, col)
        x = GAP + j*(CELL+GAP)
        draw.text(
            (x+5, 18), label,
            fill="#1B2838", font=FONT,
        )

    for i, r in enumerate(items):
        stem = r["stem"]
        row_top = HEADER + i*row_height

        draw.text(
            (GAP, row_top),
            stem,
            fill="#334155",
            font=FONT_SMALL,
        )

        with Image.open(FILES["GT"][stem]) as g:
            gt_size = g.size
            gt = g.convert("L")

        with Image.open(FILES["LR"][stem]) as f:
            rgb = f.convert("RGB").resize(
                gt_size, BILINEAR
            )

        with Image.open(FILES["depth"][stem]) as f:
            depth = f.convert("L").resize(
                gt_size, NEAREST
            )

        imgs = [rgb, depth, gt]

        for variant in VARIANTS:
            path = (
                OUT / "predictions" /
                variant / f"{stem}.png"
            )

            with Image.open(path) as f:
                p = f.convert("L")

                if p.size != gt_size:
                    raise RuntimeError(
                        f"Prediction alignment error: "
                        f"{variant}/{stem}"
                    )
                imgs.append(p.copy())

        for j, image in enumerate(imgs):
            x = GAP + j*(CELL+GAP)
            y = row_top + ROW_LABEL

            background = (
                "#FFFFFF" if j == 0 else "#000000"
            )
            tile = display_tile(
                image, CELL, background
            )
            canvas.paste(tile, (x, y))

    path = OUT / f"Fig2_{group_name}.png"
    canvas.save(path)

    print("Saved:", path)

for group, items in selected.items():
    compose(group, items)

# Balanced view: successes AND failures.
balanced = (
    selected["full_improved"][:2]
    + selected["full_regressed"][:2]
    + selected["pgdum_better_than_full"][:2]
)

# Remove duplicates while preserving order.
seen = set()
balanced = [
    r for r in balanced
    if not (r["stem"] in seen or seen.add(r["stem"]))
]

compose("balanced_review", balanced)

(OUT / "selected_samples.json").write_text(
    json.dumps(
        {
            group: [r["stem"] for r in items]
            for group, items in selected.items()
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("\nALL DONE")
print("Output:", OUT)
