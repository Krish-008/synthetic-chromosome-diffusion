
import sys, torch, numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
W, CKPT = "/users/barumugam/judge_eval/stable_diffusion", sys.argv[1]
pipe = StableDiffusionPipeline.from_single_file(f"{W}/SD_Model/v1-5-pruned-emaonly.safetensors",
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(CKPT)
pipe.set_progress_bar_config(disable=True)

G = {str(i): g for g, r in {"A": range(1,4), "B": range(4,6), "C": range(6,13), "D": range(13,16),
     "E": range(16,19), "F": range(19,21), "G": range(21,23)}.items() for i in r}
def prompt(c):
    n = f"{c} chromosome, sex chromosome" if c in ("X","Y") else f"chromosome {c}, group {G[c]}"
    return f"a G-banded human metaphase chromosome, {n}, grayscale brightfield microscopy"

classes, scales = ["1", "9", "16", "21", "X", "Y"], [1.5, 3.0, 4.5, 6.0, 7.5]
rows = []
for gs in scales:
    tiles = []
    for c in classes:
        im = pipe(prompt(c), num_inference_steps=25, guidance_scale=gs,
                  generator=torch.Generator("cuda").manual_seed(7)).images[0]
        tiles.append(np.asarray(im.convert("L").resize((160, 160))))
    rows.append(np.hstack(tiles)); print("guidance", gs, "done", flush=True)
Image.fromarray(np.vstack(rows)).save(f"{W}/guidance_sweep_{CKPT.split('_')[-1]}.png")
