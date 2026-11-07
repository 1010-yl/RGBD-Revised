import json
import time
import statistics
from pathlib import Path
from collections import defaultdict

import torch
from models.RGBDRevised import RGBDRevisedNet
from data_rgbd import RGBDDataset

ROOT = Path("/root/UMF/RGBD_revised_offline_v1")
TEST = Path("/root/UMF/NJU2000_split_revised/test")
DEVICE = "cuda"
VARIANT = "full"
BRANCHES = ["RGB-only", "Light Fusion", "Enhanced Fusion"]

torch.set_grad_enabled(False)

model = RGBDRevisedNet(
    use_pgdum=True,
    use_dapf=True,
    use_rgb_prior=True
).to(DEVICE).eval()

ckpt = torch.load(
    ROOT / "runs/formal_full_s42/best.pth",
    map_location="cpu",
    weights_only=False
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)

dataset = RGBDDataset(
    TEST,
    side=384,
    augment=False,
    corrupt_depth=False
)

routers = [
    (name, module)
    for name, module in model.named_modules()
    if name.split(".")[-1].lower() == "router"
]

print("Detected routers:", [name for name, _ in routers])

if len(routers) != 2:
    print("Router candidates:")
    for name, module in model.named_modules():
        if "route" in name.lower():
            print(name, type(module).__name__)
    raise RuntimeError(
        "Expected two router modules. "
        "Please inspect the actual model structure."
    )

counts = defaultdict(lambda: [0, 0, 0])

def make_hook(name):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"{name}: router output is not a tensor")

        scores = output.detach().reshape(output.shape[0], -1)

        if scores.shape[1] != 3:
            raise RuntimeError(
                f"{name}: expected 3 branches, got {tuple(scores.shape)}"
            )

        for idx in scores.argmax(dim=1).cpu().tolist():
            counts[name][idx] += 1

    return hook

handles = [
    module.register_forward_hook(make_hook(name))
    for name, module in routers
]

with torch.inference_mode():
    for i in range(len(dataset)):
        sample = dataset[i]

        rgb = sample["rgb"].unsqueeze(0).to(DEVICE)
        depth = sample["depth"].unsqueeze(0).to(DEVICE)
        valid = sample["valid"].unsqueeze(0).to(DEVICE)

        model(rgb, depth, valid)

        if (i + 1) % 100 == 0:
            print(f"Routing checked: {i+1}/{len(dataset)}", flush=True)

for handle in handles:
    handle.remove()

print("\n========== ROUTING STATISTICS ==========")

result = {
    "model": VARIANT,
    "num_images": len(dataset),
    "stages": {}
}

for name, values in counts.items():
    total = sum(values)
    result["stages"][name] = {}

    print(f"\n{name}, total decisions={total}")

    for branch, n in zip(BRANCHES, values):
        pct = 100 * n / total if total else 0
        print(f"  {branch:18s} {n:4d}  {pct:6.2f}%")

        result["stages"][name][branch] = {
            "count": n,
            "percentage": pct
        }

def benchmark(forced_branch=None, repeats=40):
    force_handles = []

    if forced_branch is not None:
        def force_route(module, inputs, output):
            forced = torch.full_like(output, -1000.0)
            forced[:, forced_branch] = 1000.0
            return forced

        force_handles = [
            module.register_forward_hook(force_route)
            for _, module in routers
        ]

    sample = dataset[0]
    rgb = sample["rgb"].unsqueeze(0).to(DEVICE)
    depth = sample["depth"].unsqueeze(0).to(DEVICE)
    valid = sample["valid"].unsqueeze(0).to(DEVICE)

    timings = []

    with torch.inference_mode():
        for _ in range(10):
            model(rgb, depth, valid)

        torch.cuda.synchronize()

        for _ in range(repeats):
            t0 = time.perf_counter()
            model(rgb, depth, valid)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000)

    for handle in force_handles:
        handle.remove()

    return statistics.median(timings)

print("\n========== INFERENCE LATENCY ==========")

result["latency_ms"] = {}
result["latency_ms"]["dynamic"] = benchmark()

print(f"Dynamic routing: {result['latency_ms']['dynamic']:.3f} ms")

for k, branch in enumerate(BRANCHES):
    ms = benchmark(forced_branch=k)
    result["latency_ms"][branch] = ms
    print(f"Force {branch:18s}: {ms:.3f} ms")

out = ROOT / "runs/dapf_route_audit.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("\nSaved:", out)
