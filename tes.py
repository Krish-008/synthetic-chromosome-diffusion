import json, random
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from IPython.display import Image, display

WORK = Path("/users/barumugam/judge_eval/stable_diffusion")

rows = [json.loads(l) for l in open(WORK / "sd_train_v3" / "metadata.jsonl")]
pick = random.Random(7).sample(rows, 16)
tiles = [np.asarray(PILImage.open(WORK / "sd_train_v3" / r["file_name"]).convert("L").resize((160, 160)))
         for r in pick]
grid = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 16, 8)])
PILImage.fromarray(grid).save(WORK / "train_sample.png")
display(Image(str(WORK / "train_sample.png")))
print([r["cls"] for r in pick])

