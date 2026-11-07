import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageDraw, ImageFont

from models.RGBDRevised import RGBDRevisedNet
from data_rgbd import RGBDDataset

ROOT = Path("/root/UMF/RGBD_revised_offline_v1")
TEST = Path("/root/UMF/NJU2000_split_revised/test")
SOURCE = ROOT / "runs/fig2_analysis_v2"
DEST = SOURCE / "route_four_cases"
DEST.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "001082_left",
    "001080_left",
    "001601_left",
    "001857_left",
]

BRANCHES = ["RGB-only", "Light Fusion", "Enhanced Fusion"]
MODEL_NAMES = [
    "baseline", "pgdum", "dapf",
    "full", "no_rgb_prior"
]

# ----------------------------------------------------
# 1. Load model and exact test dataset.
# ----------------------------------------------------
dataset = RGBDDataset(
    TEST, side=384, augment=False, corrupt_depth=False
)

lookup = {stem: i for i, stem in enumerate(dataset.stems)}
for stem in SAMPLES:
    if stem not in lookup:
        raise RuntimeError(f"Test sample missing: {stem}")

ckpt_path = ROOT / "runs/formal_full_s42/best.pth"

ckpt = torch.load(
    ckpt_path, map_location="cpu", weights_only=False
)

if ckpt.get("config", {}).get("variant") != "full":
    raise RuntimeError("Expected a Full checkpoint")

model = RGBDRevisedNet(
    use_pgdum=True,
    use_dapf=True,
    use_rgb_prior=True
).cuda().eval()

model.load_state_dict(
    ckpt["model_state_dict"], strict=True
)

modules = dict(model.named_modules())
router_names = {
    "Stage 2": "fuse2.router",
    "Stage 3": "fuse3.router",
}

for name in router_names.values():
    if name not in modules:
        raise RuntimeError(
            f"Missing router {name}. Found: "
            + str([
                k for k in modules
                if "router" in k.lower()
            ])
        )

# ----------------------------------------------------
# 2. Record actual eval-time router outputs.
# ----------------------------------------------------
captured = {}

def create_hook(stage):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{stage}: unsupported router output")

        score = output.detach().flatten(1)[0].float().cpu()

        if score.numel() != 3:
            raise RuntimeError(
                f"{stage}: expected three branches"
            )

        # Some routers return logits, others return
        # already-normalized softmax probabilities.
        is_probability = (
            bool(torch.all(score >= 0))
            and bool(torch.all(score <= 1))
            and abs(float(score.sum()) - 1) < 1e-4
        )

        prob = score if is_probability else torch.softmax(score, 0)
        selected = int(prob.argmax())

        captured[stage] = {
            "choice_index": selected,
            "choice": BRANCHES[selected],
            "probabilities": {
                b: float(prob[i])
                for i, b in enumerate(BRANCHES)
            }
        }

    return hook

handles = [
    modules[module_name].register_forward_hook(
        create_hook(stage)
    )
    for stage, module_name in router_names.items()
]

def find_file(folder, stem):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        path = TEST / folder / (stem + ext)
        if path.is_file():
            return path
    raise FileNotFoundError(f"{folder}/{stem}")

def read_image(folder, stem, mode):
    with Image.open(find_file(folder, stem)) as im:
        return im.convert(mode).copy()

def predicted_image(model_name, stem):
    path = (
        SOURCE / "predictions" /
        model_name / f"{stem}.png"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing prediction: {path}"
        )
    with Image.open(path) as im:
        return im.convert("L").copy()

def extract_logits(output, target_size):
    if not isinstance(output, dict):
        raise RuntimeError("Unexpected network output")

    candidates = [
        "saliency_logits", "sal_logits",
        "saliency_preds", "saliency",
        "sal", "sals"
    ]
    heads = next(
        (output[k] for k in candidates if k in output),
        None
    )

    if not isinstance(heads, (tuple, list)) or len(heads) != 4:
        raise RuntimeError(
            f"Cannot identify four saliency heads: {list(output)}"
        )

    return sum(
        F.interpolate(
            h, size=target_size,
            mode="bilinear", align_corners=False
        )
        for h in heads
    )

# Existing per-image scores from previous evaluation.
metric_path = SOURCE / "per_image_metrics.csv"
with metric_path.open(
    newline="", encoding="utf-8-sig"
) as f:
    metrics = {
        row["stem"]: row
        for row in csv.DictReader(f)
    }

results = []

with torch.inference_mode():
    for stem in SAMPLES:
        captured.clear()

        sample = dataset[lookup[stem]]

        rgb_t = sample["rgb"].unsqueeze(0).cuda()
        dep_t = sample["depth"].unsqueeze(0).cuda()
        val_t = sample["valid"].unsqueeze(0).cuda()

        out = model(rgb_t, dep_t, val_t)

        if set(captured) != set(router_names):
            raise RuntimeError(
                f"{stem}: incomplete routing capture: {captured}"
            )

        # Confirm that the current forward pass matches
        # the earlier saved Full prediction.
        gt = read_image("GT", stem, "L")
        logits = extract_logits(out, rgb_t.shape[-2:])

        native = F.interpolate(
            logits, size=(gt.height, gt.width),
            mode="bilinear", align_corners=False
        )

        new_pred = np.clip(
            np.rint(
                native.sigmoid()[0, 0].cpu().numpy() * 255
            ), 0, 255
        ).astype(np.uint8)

        old_pred = np.asarray(
            predicted_image("full", stem),
            dtype=np.uint8
        )

        if new_pred.shape != old_pred.shape:
            raise RuntimeError(
                f"{stem}: prediction size mismatch"
            )

        max_difference = int(
            np.abs(
                new_pred.astype(np.int16)
                - old_pred.astype(np.int16)
            ).max()
        )

        if max_difference > 2:
            raise RuntimeError(
                f"{stem}: saved Full prediction differs "
                f"from current forward pass; max pixel "
                f"difference={max_difference}. "
                f"Do not link this route to that prediction."
            )

        m = metrics[stem]

        record = {
            "stem": stem,
            "routes": dict(captured),
            "metrics": {
                name: {
                    "MAE": float(m[f"{name}_MAE"]),
                    "Fw": float(m[f"{name}_Fw"]),
                }
                for name in MODEL_NAMES
            },
            "prediction_verification_max_u8_difference":
                max_difference,
        }

        results.append(record)

        print(f"\n===== {stem} =====")
        for stage, route in captured.items():
            print(stage, "->", route["choice"])
            print("Probabilities:", route["probabilities"])

        print(
            "Baseline/PG-DUM/Full MAE:",
            *[
                f"{record['metrics'][name]['MAE']:.5f}"
                for name in ["baseline", "pgdum", "full"]
            ]
        )

for h in handles:
    h.remove()

(DEST / "routes_four_cases.json").write_text(
    json.dumps(
        results, ensure_ascii=False, indent=2
    ),
    encoding="utf-8"
)

rows = []

for record in results:
    row = {"stem": record["stem"]}

    for stage in router_names:
        prefix = stage.lower().replace(" ", "_")
        info = record["routes"][stage]
        row[f"{prefix}_selected"] = info["choice"]

        for branch in BRANCHES:
            col = branch.lower().replace(" ", "_").replace("-", "_")
            row[f"{prefix}_p_{col}"] = info["probabilities"][branch]

    for name in MODEL_NAMES:
        row[f"{name}_MAE"] = record["metrics"][name]["MAE"]
        row[f"{name}_Fw"] = record["metrics"][name]["Fw"]

    rows.append(row)

with (DEST / "routes_four_cases.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

# ----------------------------------------------------
# 3. Create error maps and prediction-difference maps.
# ----------------------------------------------------
# White = true positive
# Red   = false positive
# Blue  = false negative
# Black = true negative

def error_map(pred, gt):
    p = np.asarray(pred, dtype=np.uint8) >= 128
    g = np.asarray(gt, dtype=np.uint8) >= 128

    result = np.zeros(
        (g.shape[0], g.shape[1], 3),
        dtype=np.uint8
    )

    result[p & g] = (255, 255, 255)
    result[p & ~g] = (245, 65, 65)
    result[~p & g] = (40, 150, 245)

    return Image.fromarray(result, "RGB")

def difference_map(a, b):
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)

    diff = np.abs(aa - bb)

    result = np.zeros(
        (diff.shape[0], diff.shape[1], 3),
        dtype=np.uint8
    )

    result[..., 0] = np.clip(diff * 2, 0, 255)
    result[..., 1] = np.clip(diff * 0.5, 0, 255)

    return Image.fromarray(result, "RGB")

COLUMNS = [
    "RGB", "Depth", "GT", "Baseline",
    "PG-DUM", "DAPF", "Full", "w/o Prior",
    "Error Base", "Error PG", "Error Full",
    "|Full-Base|", "|Full-PG|"
]

CELL, GAP, ROW_HEAD = 188, 9, 51
TOP = 65

width = GAP + len(COLUMNS) * (CELL + GAP)
height = TOP + len(SAMPLES) * (CELL + ROW_HEAD + GAP) + GAP

canvas = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        17
    )
    small = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        14
    )
except OSError:
    font = ImageFont.load_default()
    small = font

for j, label in enumerate(COLUMNS):
    draw.text(
        (GAP + j * (CELL + GAP), 18),
        label, fill="#233044", font=font
    )

def tile(image, background, method):
    fitted = ImageOps.contain(
        image.convert("RGB"),
        (CELL, CELL),
        method=method
    )
    result = Image.new(
        "RGB", (CELL, CELL), background
    )
    result.paste(
        fitted,
        (
            (CELL - fitted.width) // 2,
            (CELL - fitted.height) // 2
        )
    )
    return result

for i, record in enumerate(results):
    stem = record["stem"]
    row_y = TOP + i * (CELL + ROW_HEAD + GAP)

    s2 = record["routes"]["Stage 2"]["choice"]
    s3 = record["routes"]["Stage 3"]["choice"]

    label = (
        f"{stem}     Stage 2: {s2}"
        f"     Stage 3: {s3}"
    )

    draw.text(
        (GAP, row_y), label,
        fill="#26364B", font=font
    )

    gt = read_image("GT", stem, "L")
    target = gt.size

    rgb = read_image(
        "LR", stem, "RGB"
    ).resize(target, Image.Resampling.BILINEAR)

    depth = read_image(
        "depth", stem, "L"
    ).resize(target, Image.Resampling.NEAREST)

    predictions = {
        name: predicted_image(name, stem)
        for name in MODEL_NAMES
    }

    images = [
        rgb, depth, gt,
        predictions["baseline"],
        predictions["pgdum"],
        predictions["dapf"],
        predictions["full"],
        predictions["no_rgb_prior"],
        error_map(predictions["baseline"], gt),
        error_map(predictions["pgdum"], gt),
        error_map(predictions["full"], gt),
        difference_map(
            predictions["full"],
            predictions["baseline"]
        ),
        difference_map(
            predictions["full"],
            predictions["pgdum"]
        ),
    ]

    for j, img in enumerate(images):
        if img.size != gt.size:
            raise RuntimeError(
                f"{stem}: size mismatch, column {j}"
            )

        background = (
            "white" if j == 0 else "black"
        )
        method = (
            Image.Resampling.NEAREST
            if j in (8, 9, 10)
            else Image.Resampling.BILINEAR
        )

        canvas.paste(
            tile(img, background, method),
            (
                GAP + j * (CELL + GAP),
                row_y + ROW_HEAD
            )
        )

out_path = DEST / "Fig2_routes_and_errors.png"
canvas.save(out_path)

print("\nSaved routes and error figure:")
print(out_path)
print("Red = false positive")
print("Blue = false negative")
print("White = correctly detected foreground")
print("Orange intensity = prediction difference")
