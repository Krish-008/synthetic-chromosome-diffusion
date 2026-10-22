"""
24-class chromosome classifier (ResNet-18) trained on REAL crops, used as a judge
for generated images.

Real crops are rendered with exactly the v4 canvas recipe (bg 245, upscale <= 4x,
centred on 512 canvas), so the classifier sees real and generated images in the
same format, including relative size.

Split: by source metaphase image (crops.csv `src`), so chromosomes from the same
spread never appear in both train and test.

Outputs (in --out):
  best.pt            model weights (highest test class accuracy)
  test_report.csv    per-class accuracy on held-out real crops
  confusion.csv      24x24 confusion matrix (rows = true, cols = predicted)
"""
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

W = Path("/users/barumugam/judge_eval/stable_diffusion")
p = argparse.ArgumentParser()
p.add_argument("--crops", default=str(W / "crops_v1"))
p.add_argument("--out", default=str(W / "classifier_v1"))
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--size", type=int, default=224)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--test_frac", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--limit", type=int, default=0, help="debug: max crops per class")
args = p.parse_args()

CROPS, OUT = Path(args.crops), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

ORDER = [str(i) for i in range(1, 23)] + ["X", "Y"]
CLS2IDX = {c: i for i, c in enumerate(ORDER)}
GROUP = {**{str(i): g for g, r in {"A": range(1, 4), "B": range(4, 6), "C": range(6, 13),
                                   "D": range(13, 16), "E": range(16, 19), "F": range(19, 21),
                                   "G": range(21, 23)}.items() for i in r}, "X": "C", "Y": "G"}
GROUP_OF_IDX = np.array([GROUP[c] for c in ORDER])

# ---- v4 canvas recipe (must match make_canvas_v4.py) ----
CANVAS, FILL, MAX_SCALE, BG = 512, 0.80, 4.0, 245


def to_canvas(img, mask):
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img, mask = img[y0:y1, x0:x1].astype(np.float32), mask[y0:y1, x0:x1]
    bg = np.median(img[mask == 0]) if (mask == 0).any() else img.max()
    img = np.clip(img * (BG / max(bg, 1)), 0, BG)
    s = min(FILL * CANVAS / max(y1 - y0, x1 - x0), MAX_SCALE)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    im = np.clip(cv2.resize(img, None, fx=s, fy=s, interpolation=interp), 0, BG)
    hard = (cv2.resize(mask, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR) > 127).astype(np.float32)
    alpha = cv2.GaussianBlur(hard, (0, 0), sigmaX=1.5)
    canvas = np.full((CANVAS, CANVAS), BG, np.float32)
    y, x = (CANVAS - im.shape[0]) // 2, (CANVAS - im.shape[1]) // 2
    sl = (slice(y, y + im.shape[0]), slice(x, x + im.shape[1]))
    canvas[sl] = alpha * im + (1 - alpha) * canvas[sl]
    return canvas


def to_tensor(gray512, size):
    """512 grayscale canvas (0-255) -> normalized 1x size x size tensor. Use for generated images too."""
    g = cv2.resize(gray512.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy((g / 255.0 - 0.5) / 0.5)[None]


class RealCrops(Dataset):
    def __init__(self, df, train):
        self.df, self.train = df.reset_index(drop=True), train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(CROPS / "img" / r.file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(CROPS / "mask" / r.file), cv2.IMREAD_GRAYSCALE)
        if self.train:                               # orientation is arbitrary in metaphase spreads
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k).copy(), np.rot90(mask, k).copy()
            if random.random() < 0.5:
                img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        return to_tensor(to_canvas(img, mask), args.size), CLS2IDX[r.cls]


def build_model():
    m = resnet18(weights=None)
    try:                                             # ImageNet init if weights are reachable/cached
        from torchvision.models import ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        print("init: ImageNet weights", flush=True)
    except Exception as e:
        print(f"init: random (ImageNet weights unavailable: {type(e).__name__})", flush=True)
    w = m.conv1.weight.data.sum(1, keepdim=True)     # 3-channel -> 1-channel stem
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False); m.conv1.weight.data = w
    m.fc = nn.Linear(m.fc.in_features, len(ORDER))
    return m


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        with torch.autocast("cuda", enabled=dev == "cuda"):
            y_pred.append(model(x.to(dev)).argmax(1).cpu()); y_true.append(y)
    y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
    cls_acc = (y_true == y_pred).mean()
    grp_acc = (GROUP_OF_IDX[y_true] == GROUP_OF_IDX[y_pred]).mean()
    return cls_acc, grp_acc, y_true, y_pred


# ---- data: kept crops, split by source metaphase image ----
df = pd.read_csv(CROPS / "crops.csv")
df = df[(df.status == "kept") & df.file.notna()].copy()
df["cls"] = df["cls"].astype(str)
df = df[df.cls.isin(ORDER)]
if args.limit:
    df = df.groupby("cls", group_keys=False).apply(lambda g: g.sample(min(len(g), args.limit), random_state=0))
srcs = sorted(df.src.unique()); random.Random(args.seed).shuffle(srcs)
test_src = set(srcs[:int(len(srcs) * args.test_frac)])
tr, te = df[~df.src.isin(test_src)], df[df.src.isin(test_src)]
print(f"train {len(tr)} crops / {tr.src.nunique()} spreads | test {len(te)} crops / {te.src.nunique()} spreads",
      flush=True)
pd.Series(sorted(test_src)).to_csv(OUT / "test_spreads.csv", index=False, header=["src"])

dl_tr = DataLoader(RealCrops(tr, True), args.batch, shuffle=True, num_workers=args.workers,
                   drop_last=True, pin_memory=True, persistent_workers=args.workers > 0)
dl_te = DataLoader(RealCrops(te, False), args.batch, num_workers=args.workers, pin_memory=True)

# ---- train ----
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl_tr))
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
best, T0 = -1.0, time.time()

for ep in range(1, args.epochs + 1):
    model.train(); run = 0.0
    for x, y in dl_tr:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", enabled=dev == "cuda"):
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
    cls_acc, grp_acc, yt, yp = evaluate(model, dl_te, dev)
    print(f"epoch {ep}  loss {run / len(dl_tr):.3f}  test class acc {cls_acc:.3f}  "
          f"group acc {grp_acc:.3f}  ({(time.time() - T0) / 60:.1f} min)", flush=True)
    if cls_acc > best:
        best = cls_acc
        torch.save({"model": model.state_dict(), "classes": ORDER, "size": args.size}, OUT / "best.pt")
        cm = np.zeros((len(ORDER), len(ORDER)), int)
        np.add.at(cm, (yt, yp), 1)
        pd.DataFrame(cm, index=ORDER, columns=ORDER).to_csv(OUT / "confusion.csv")
        per = pd.DataFrame({"cls": ORDER, "n": cm.sum(1),
                            "acc": np.diag(cm) / np.maximum(cm.sum(1), 1)}).round(3)
        per.to_csv(OUT / "test_report.csv", index=False)

print(f"best test class acc {best:.3f} -> {OUT / 'best.pt'}", flush=True)
print(pd.read_csv(OUT / "test_report.csv").to_string(index=False))
