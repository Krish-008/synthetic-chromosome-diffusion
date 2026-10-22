"""
Score a LoRA with the real-crop ResNet-18 judge.

For each of the 24 classes: generate N images with the class caption, classify them,
and report class accuracy (exact chromosome) and group accuracy (A-G).

Outputs (in --out):
  report.csv      per-class class_acc / group_acc / most-common wrong prediction
  confusion.csv   24x24, rows = requested class, cols = judge prediction
  grid.png        first 6 samples per class (rows = classes 1..22, X, Y)
  images/         all generated images (for KID/FID later)

Images are background-normalised before judging (border median -> 245, clipped at 245,
the training-canvas rule). `valid` = looks like one chromosome on a plain background:
plain border, exactly one dark object, not touching the edge, plausible size.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from torchvision.models import resnet18

W = Path("/users/barumugam/judge_eval/stable_diffusion")
p = argparse.ArgumentParser()
p.add_argument("--lora", default=str(W / "lora_v3" / "final"))
p.add_argument("--judge", default=str(W / "classifier_v1" / "best.pt"))
p.add_argument("--out", default=str(W / "eval_lora_v3"))
p.add_argument("--caption", choices=["v3", "v4"], default="v3", help="must match how the LoRA was trained")
p.add_argument("--n", type=int, default=50, help="images per class")
p.add_argument("--batch", type=int, default=10)
p.add_argument("--guidance", type=float, default=4.5)
p.add_argument("--steps", type=int, default=25)
p.add_argument("--seed", type=int, default=1000)
args = p.parse_args()

OUT = Path(args.out); (OUT / "images").mkdir(parents=True, exist_ok=True)
dev, fp16 = "cuda", torch.float16

ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]
GROUP = {**{str(i): g for g, r in {"A": range(1, 4), "B": range(4, 6), "C": range(6, 13),
                                   "D": range(13, 16), "E": range(16, 19), "F": range(19, 21),
                                   "G": range(21, 23)}.items() for i in r}, "X": "C", "Y": "G"}


def caption(c):
    if args.caption == "v4":
        return f"a G-banded human metaphase chromosome, chromosome {c}, grayscale brightfield microscopy"
    name = f"{c} chromosome, sex chromosome" if c in ("X", "Y") else f"chromosome {c}, group {GROUP[c]}"
    return f"a G-banded human metaphase chromosome, {name}, grayscale brightfield microscopy"


# ---- judge (same preprocessing as train_classifier.py) ----
ck = torch.load(args.judge, map_location="cpu", weights_only=False)
judge = resnet18(weights=None)
judge.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
judge.fc = nn.Linear(judge.fc.in_features, len(ORDER))
judge.load_state_dict(ck["model"]); judge.eval().to(dev)
SIZE = ck["size"]
assert ck["classes"] == ORDER


BG, BORDER = 245, 16


def border_px(g):
    return np.concatenate([g[:BORDER].ravel(), g[-BORDER:].ravel(), g[:, :BORDER].ravel(), g[:, -BORDER:].ravel()])


def normalise(g):
    return np.clip(g.astype(np.float32) * (BG / max(float(np.median(border_px(g))), 1)), 0, BG)


def is_chromosome(n):
    """n: background-normalised image. True if one chromosome-like object on a plain background."""
    if border_px(n).std() > 12:                              # textured / photographic background
        return False
    dark = cv2.morphologyEx((n < 0.8 * BG).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    big = [i for i in range(1, k) if stats[i, cv2.CC_STAT_AREA] >= 200]
    if len(big) != 1:                                        # nothing, or several objects / letters
        return False
    x, y, w, h, a = stats[big[0]]
    if x <= 4 or y <= 4 or x + w >= n.shape[1] - 4 or y + h >= n.shape[0] - 4:
        return False                                         # touches the edge
    return a >= 0.9 * dark.sum() and 500 <= a <= 0.3 * n.size


def to_tensor(gray512):
    g = cv2.resize(gray512.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((g / 255.0 - 0.5) / 0.5)[None]


@torch.no_grad()
def classify(grays):
    x = torch.stack([to_tensor(g) for g in grays]).to(dev)
    return judge(x).argmax(1).cpu().numpy()


# ---- generator ----
pipe = StableDiffusionPipeline.from_single_file(str(W / "SD_Model" / "v1-5-pruned-emaonly.safetensors"),
                                                torch_dtype=fp16, safety_checker=None,
                                                requires_safety_checker=False).to(dev)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(args.lora)
pipe.set_progress_bar_config(disable=True)

cm = np.zeros((len(ORDER), len(ORDER)), int)
valid_n, valid_hits = np.zeros(len(ORDER), int), np.zeros(len(ORDER), int)
grid_rows = []
for ci, c in enumerate(ORDER):
    grays = []
    for b in range(0, args.n, args.batch):
        k = min(args.batch, args.n - b)
        gens = [torch.Generator(dev).manual_seed(args.seed + ci * 10_000 + b + j) for j in range(k)]
        ims = pipe([caption(c)] * k, num_inference_steps=args.steps, guidance_scale=args.guidance,
                   generator=gens).images
        for j, im in enumerate(ims):
            g = np.asarray(im.convert("L"))
            cv2.imwrite(str(OUT / "images" / f"{c}_{b + j:03d}.png"), g)
            grays.append(g)
    norm = [normalise(g) for g in grays]
    valid = np.array([is_chromosome(n) for n in norm])
    pred = classify(norm)
    np.add.at(cm, (np.full_like(pred, ci), pred), 1)
    valid_n[ci], valid_hits[ci] = valid.sum(), (pred[valid] == ci).sum()
    grid_rows.append(np.hstack([cv2.resize(g, (128, 128)) for g in grays[:6]]))
    print(f"chr {c:>2}: valid {valid.mean():.2f}  class acc {np.mean(pred == ci):.2f}  "
          f"group acc {np.mean([GROUP[ORDER[q]] == GROUP[c] for q in pred]):.2f}", flush=True)

# ---- report ----
grp = np.array([GROUP[c] for c in ORDER])
rows = []
for ci, c in enumerate(ORDER):
    r = cm[ci]
    wrong = r.copy(); wrong[ci] = 0
    rows.append({"cls": c, "n": r.sum(), "valid_rate": valid_n[ci] / r.sum(),
                 "class_acc": r[ci] / r.sum(),
                 "class_acc_valid": valid_hits[ci] / max(valid_n[ci], 1),
                 "group_acc": r[grp == grp[ci]].sum() / r.sum(),
                 "top_wrong": ORDER[wrong.argmax()] if wrong.sum() else "-"})
rep = pd.DataFrame(rows).round(3)
rep.to_csv(OUT / "report.csv", index=False)
pd.DataFrame(cm, index=ORDER, columns=ORDER).to_csv(OUT / "confusion.csv")
cv2.imwrite(str(OUT / "grid.png"), np.vstack(grid_rows))

total = cm.sum()
print(rep.to_string(index=False))
print(f"\nOVERALL valid {valid_n.sum() / total:.3f} | class acc {np.trace(cm) / total:.3f} | group acc "
      f"{sum(cm[i][grp == grp[i]].sum() for i in range(len(ORDER))) / total:.3f} | "
      f"class acc on valid {valid_hits.sum() / max(valid_n.sum(), 1):.3f} -> {OUT}")
