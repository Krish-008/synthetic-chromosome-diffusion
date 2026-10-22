**synthetic-chromosome-diffusion**

Synthetic chromosome generation for karyotype segmentation.

Fine-tuning Stable Diffusion v1.5 to generate G-banded chromosome images, with a quantitative evaluation pipeline instead of visual inspection.

Status: in progress. Two model iterations complete; overlap compositing and the downstream segmentation experiment are not yet done.

Why

Automated karyotyping needs to separate chromosomes that overlap in metaphase spreads. Inside a dense overlap, per-pixel assignment is partly interpretation, so annotation is unreliable, and for clusters of three or more chromosomes annotated data does not exist at all. The benchmark most commonly used for overlap segmentation is built by copy-pasting overlaps drawn from a pool of only 46 source chromosomes.

The approach here is to generate the data instead, so that the per-pixel structure is known by construction rather than inferred from an annotator's judgement.

Data

Single-chromosome crops with masks from the Tseng et al. G-banded dataset (~83,000 crops), filtered and resampled into 19,200 class-balanced canvases. The dataset is not redistributed here; see the original source for access and licensing.

Method
Stable Diffusion v1.5, fine-tuned with LoRA on the UNet and the CLIP text encoder, so that generation is conditioned on chromosome identity.
Trained on a single V100 GPU on the SMU high-performance computing cluster.

Before fine-tuning, the base model given a chromosome prompt produced generic microscopy texture with no banding, centromere or arm structure (zero-shot outputs).

Evaluation

Visual grids are not evidence, so generation is scored automatically:

Identity classifier. ResNet-18 trained on real crops, ~99% accuracy on held-out spreads.
Validity check. A structural filter (one object on a plain background, not touching the edge, plausible size) that 97% of real chromosomes pass, applied to generated images across 50 seeds per class rather than a single grid.
Reported metrics. Validity rate, group accuracy (chromosome group A–G), class accuracy (individual chromosome).
Results

1,200 generated images per model (50 per class, 24 classes).

Model	Valid (across seeds)	Group accuracy	Class accuracy
Model 1: rank-16 LoRA, attention layers	61%	28%	10%
Model 2: rank-64 LoRA, attention + convolution	89%	50%	19%
Real chromosomes (reference)	97%	—	~99%

Per-class reports and confusion matrices (rows = requested chromosome, columns = classifier's prediction):

Model 1: report.csv · confusion.csv
Model 2: report.csv · confusion.csv

Model 1 looked acceptable on a single-seed grid and failed under measurement. It leaked the base model (wood textures, fragments of text), tied background brightness to chromosome size, and learned size from a shortcut in the caption template rather than from chromosome identity. Its failures concentrate in the large chromosomes: only 8–14% of images for chromosomes 1–3 are valid, and the classifier labels most of chromosomes 2–15 as 1 or 4.

Model 2 addressed those causes rather than training longer: a single caption template with the group shortcut removed, offset noise, and a higher-rank LoRA extended to the convolution layers. Backgrounds are clean and the base-model leakage is gone; validity for chromosomes 1–3 rises to 76–86%, and X and Y reach 100%.

Model 1	Model 2
Show Image	Show Image
Current limitations
Banding patterns are not reproduced, so chromosomes within the same group are not yet distinguishable. In Model 2's confusion matrix, group C chromosomes (6–12) are spread across the whole group, and chromosomes 19–22 are most often labelled 22.
Class accuracy at 19% is far from usable; group-level structure is learned, individual identity largely is not.
Y scores 86%, but generated Y chromosomes are small dark shapes without banding; the classifier likely keys on size and darkness, so this is not evidence of realistic Y chromosomes.
No overlap data has been generated yet, and the downstream segmentation experiment has not been run.
Next steps
Full UNet fine-tuning and a from-scratch baseline, to establish whether the limit is model capacity or the VAE's latent compression.
Compositing overlaps directly, so the segmentation mask is known by construction, using the generative model to make the composite resemble real microscopy rather than to produce the labels.
The downstream test, fixed in advance: a segmentation model trained on generated overlaps must outperform the same model trained on copy-paste overlaps, evaluated on real clusters. Fidelity metrics alone do not establish that synthetic data is fit for purpose.
Repository contents
File / folder	What it is
train_lora.py	Fine-tuning script, Model 1
train_lora_v4.py	Fine-tuning script, Model 2 (new captions, offset noise, rank-64 LoRA incl. convolutions)
train_classifier.py	ResNet-18 identity classifier trained on real crops
eval_generated_1.py	Generates 50 images per class and scores validity, group and class accuracy
sweep.py	Guidance-scale sweep for a checkpoint
Images before FT/	Base-model outputs before fine-tuning
Information-after_first-fine-tuning/	Model 1 results: per-class report, confusion matrix, samples
Information_after_second_fine_tuning/	Model 2 results: per-class report, confusion matrix, samples

The scripts currently use absolute paths from the HPC project folder; the base model, dataset and canvases are not included.

Licence

Model weights derived from Stable Diffusion v1.5 are subject to the CreativeML OpenRAIL-M licence. Research use only; not for clinical or diagnostic use.

Author

Balakrishnan Arumugam — MS Computer Science, Southern Methodist University balakrishnanarumugam08@gmail.com · github.com/Krish-008
