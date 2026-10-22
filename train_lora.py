"""
LoRA fine-tuning of Stable Diffusion v1.5 (UNet + CLIP text encoder) on
single-chromosome canvases. Target: 1x V100 32GB, fp16 autocast + GradScaler.
Rerunning the same command resumes from <out>/state.pt.
"""
import argparse, csv, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="/users/barumugam/judge_eval/stable_diffusion/SD_Model")
p.add_argument("--data", default="/users/barumugam/judge_eval/stable_diffusion/sd_train_v3")
p.add_argument("--out", default="/users/barumugam/judge_eval/stable_diffusion/lora_v1")
p.add_argument("--steps", type=int, default=10000)
p.add_argument("--batch", type=int, default=4)
p.add_argument("--accum", type=int, default=2)        # effective batch = batch * accum
p.add_argument("--rank", type=int, default=16)
p.add_argument("--lr_unet", type=float, default=1e-4)
p.add_argument("--lr_te", type=float, default=5e-5)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--val_every", type=int, default=1000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_minutes", type=float, default=105)
p.add_argument("--caption_dropout", type=float, default=0.1)
args = p.parse_args()
T0 = time.time()
TO = time.time()

torch.manual_seed(args.seed); random.seed(args.seed)
dev, fp16 = "cuda", torch.float16
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

# ---- frozen base model ----
tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder",
                                   variant="fp16", dtype=fp16).to(dev)
vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=fp16).to(dev)
unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet",
                                            variant="fp16", torch_dtype=fp16).to(dev)
noise_sched = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
assert noise_sched.config.prediction_type == "epsilon"
for m in (te, vae, unet):
    m.requires_grad_(False)

# ---- LoRA adapters ----
unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                            target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
te.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                          target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
unet_params = [q for q in unet.parameters() if q.requires_grad]
te_params = [q for q in te.parameters() if q.requires_grad]
for q in unet_params + te_params:
    q.data = q.data.float()                              # trainable weights in fp32
print(f"trainable params: unet {sum(q.numel() for q in unet_params)/1e6:.2f}M, "
      f"text encoder {sum(q.numel() for q in te_params)/1e6:.2f}M")

# ---- data ----
class Canvases(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.rows = [json.loads(l) for l in open(self.root / "metadata.jsonl")]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = np.asarray(Image.open(self.root / r["file_name"]).convert("RGB"), np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1) / 127.5 - 1
        if torch.rand(()) < 0.5:
            x = x.flip(-1)
        text = "" if torch.rand(()) < args.caption_dropout else r["text"]
        ids = tok(text, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids[0]
        return x, ids

ds = Canvases(args.data)
dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2,
                drop_last=True, pin_memory=True)
def batches():
    while True:
        yield from dl

CLASSES = [str(i) for i in range(1, 23)] + ["X", "Y"]
prompts = {c: next(r["text"] for r in ds.rows if r["cls"] == c) for c in CLASSES}

# ---- optimisation ----
opt = torch.optim.AdamW([{"params": unet_params, "lr": args.lr_unet},
                         {"params": te_params, "lr": args.lr_te}], weight_decay=1e-2)
lr_sched = get_scheduler("cosine", opt, num_warmup_steps=args.warmup,
                         num_training_steps=args.steps)
scaler = torch.amp.GradScaler("cuda")

def save_lora(d):
    d.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        d,
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        text_encoder_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(te)),
        safe_serialization=True)

@torch.no_grad()
def validate(step):
    """One image per class, fixed seed, so checkpoints are directly comparable."""
    unet.eval(); te.eval()
    pipe = StableDiffusionPipeline(
        vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_config(noise_sched.config),
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    tiles = []
    with torch.autocast("cuda", dtype=fp16):
        for c in CLASSES:
            g = torch.Generator(dev).manual_seed(1234)
            im = pipe(prompts[c], num_inference_steps=25, guidance_scale=4.5,
                      generator=g).images[0]
            tiles.append(np.asarray(im.convert("L").resize((128, 128))))
    grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 24, 8)])
    Image.fromarray(grid).save(out / f"val_{step:05d}.png")
    unet.train(); te.train()

# ---- resume ----
state_path, step = out / "state.pt", 0
if state_path.exists():
    s = torch.load(state_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(unet, s["unet"]); set_peft_model_state_dict(te, s["te"])
    opt.load_state_dict(s["opt"]); lr_sched.load_state_dict(s["sched"])
    scaler.load_state_dict(s["scaler"]); step = s["step"]
    print(f"resumed at step {step}", flush=True)


def checkpoint(step):
    save_lora(out / f"ckpt_{step:05d}")
    torch.save({"step": step, "opt": opt.state_dict(), "sched": lr_sched.state_dict(),
                "scaler": scaler.state_dict(),
                "unet": get_peft_model_state_dict(unet),
                "te": get_peft_model_state_dict(te)}, state_path)


# ---- training loop ----
log = open(out / "loss.csv", "a", newline="")
writer = csv.writer(log)
unet.train(); te.train()
it = batches()
if step == 0:
    validate(0)                                          # zero-shot baseline grid

while step < args.steps:
    opt.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(args.accum):
        x, ids = next(it)
        x, ids = x.to(dev, fp16, non_blocking=True), ids.to(dev, non_blocking=True)
        with torch.no_grad():
            lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(lat)
        t = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],), device=dev)
        noisy = noise_sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=fp16):
            pred = unet(noisy, t, te(ids)[0]).sample
        loss = F.mse_loss(pred.float(), noise.float()) / args.accum
        scaler.scale(loss).backward()
        total += loss.item()

    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(unet_params + te_params, 1.0)
    scaler.step(opt); scaler.update(); lr_sched.step()
    step += 1

    if step % 50 == 0:
        writer.writerow([step, round(total, 5), lr_sched.get_last_lr()[0]]); log.flush()
        print(f"step {step:>6}  loss {total:.4f}  "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  "
              f"elapsed {(time.time() - T0) / 60:.1f} min", flush=True)

    time_up = (time.time() - T0) / 60 > args.max_minutes
    if step % args.val_every == 0 or step == args.steps or time_up:
        checkpoint(step)
        validate(step)
        if time_up:
            print(f"time budget reached at step {step}; rerun to resume", flush=True)
            raise SystemExit(0)

save_lora(out / "final")
print("done", flush=True)
