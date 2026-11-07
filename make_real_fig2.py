import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models.RGBDRevised import RGBDRevisedNet
from data_rgbd import RGBDDataset

ROOT = Path("/root/UMF/RGBD_revised_offline_v1")
TEST = Path("/root/UMF/NJU2000_split_revised/test")
OUT = ROOT / "runs/fig2_real_predictions.png"

VARIANTS = {
    "Baseline": ("baseline", False, False, True),
    "PG-DUM": ("pgdum", True, False, True),
    "DAPF": ("dapf", False, True, True),
    "Full": ("full", True, True, True),
    "w/o RGB Prior": ("no_rgb_prior", True, True, False),
}

dataset = RGBDDataset(
    TEST, side=384, augment=False, corrupt_depth=False
)

random.seed(42)
indices = sorted(random.sample(range(len(dataset)), 6))
stems = [dataset.stems[i] for i in indices]

predictions = {}

def get_saliency_logits(output, target_size):
    if not isinstance(output, dict):
        raise RuntimeError(
            "Unexpected model output type: "
            + str(type(output))
        )

    candidates = [
        "saliency_logits",
        "sal_logits",
        "saliency_preds",
        "saliency",
        "sal",
        "sals"
    ]

    value = None

    for key in candidates:
        if key in output:
            value = output[key]
            break

    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise RuntimeError(
            "Cannot identify four saliency-logit heads. "
            f"Actual output keys: {list(output.keys())}. "
            "Verify against evaluate_rgbd.py before plotting."
        )

    logits = []

    for tensor in value:
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Saliency head is not a tensor")

        tensor = F.interpolate(
            tensor, size=target_size,
            mode="bilinear", align_corners=False
        )
        logits.append(tensor)

    return torch.sigmoid(sum(logits))

for label, (variant, pgdum, dapf, prior) in VARIANTS.items():
    print("Predicting:", label, flush=True)

    model = RGBDRevisedNet(
        use_pgdum=pgdum,
        use_dapf=dapf,
        use_rgb_prior=prior
    ).cuda().eval()

    path = ROOT / f"runs/formal_{variant}_s42/best.pth"
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    predictions[label] = {}

    with torch.inference_mode():
        for idx, stem in zip(indices, stems):
            batch = dataset[idx]

            rgb = batch["rgb"].unsqueeze(0).cuda()
            depth = batch["depth"].unsqueeze(0).cuda()
            valid = batch["valid"].unsqueeze(0).cuda()

            output = model(rgb, depth, valid)
            prob = get_saliency_logits(output, rgb.shape[-2:])

            predictions[label][stem] = (
                prob[0, 0].detach().cpu().numpy()
            )

    del model
    torch.cuda.empty_cache()

def find_image(folder, stem):
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = TEST / folder / (stem + ext)
        if p.exists():
            return Image.open(p)
    raise FileNotFoundError(f"{folder}/{stem}")

def tile(img, size=210):
    img = img.convert("RGB")
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), "white")
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

def show_depth(img):
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [1, 99])

    if hi > lo:
        arr = np.clip((arr-lo)/(hi-lo), 0, 1)
    else:
        arr = np.zeros_like(arr)

    return Image.fromarray(
        (arr*255).astype(np.uint8), "L"
    )

labels = ["RGB", "Depth", "GT"] + list(VARIANTS.keys())

W = 210
PAD = 8
HEAD = 46
ROW = 218

canvas = Image.new(
    "RGB",
    (len(labels)*(W+PAD)+PAD, HEAD+len(stems)*ROW),
    "white"
)

draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15
    )
except OSError:
    font = ImageFont.load_default()

for j, label in enumerate(labels):
    draw.text((PAD+j*(W+PAD)+4, 12), label, fill="black", font=font)

for i, stem in enumerate(stems):
    rgb = find_image("LR", stem)
    depth = show_depth(find_image("depth", stem))
    gt = find_image("GT", stem).convert("L")

    columns = [rgb, depth, gt]

    for label in VARIANTS:
        arr = predictions[label][stem]
        image = Image.fromarray(
            (np.clip(arr, 0, 1)*255).astype(np.uint8), "L"
        )
        columns.append(image)

    for j, img in enumerate(columns):
        canvas.paste(
            tile(img),
            (PAD+j*(W+PAD), HEAD+i*ROW)
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT)

print("Saved:", OUT)
print("Test sample IDs:", stems)
