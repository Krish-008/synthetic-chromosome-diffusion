# synthetic-chromosome-diffusion
Synthetic chromosome generation for karyotype segmentation

Fine-tuning Stable Diffusion v1.5 to generate G-banded chromosome images, with a quantitative evaluation pipeline instead of visual inspection.

Status: in progress. Two model iterations complete; overlap compositing and the downstream segmentation experiment are not yet done.

Why

Automated karyotyping needs to separate chromosomes that overlap in metaphase spreads. Inside a dense overlap, per-pixel assignment is partly interpretation, so annotation is unreliable — and for clusters of three or more chromosomes, annotated data does not exist at all. The benchmark most commonly used for overlap segmentation is built by copy-pasting overlaps drawn from a pool of only 46 source chromosomes.

The approach here is to generate the data instead, so that the per-pixel structure is known by construction rather than inferred from an annotator's judgement.

Data

Single-chromosome crops with masks from the Tseng et al. G-banded dataset (~83,000 crops), filtered and resampled into 19,200 class-balanced canvases. The dataset is not redistributed here; see the original source for access and licensing.

Method
Stable Diffusion v1.5, fine-tuned with LoRA on the UNet and the CLIP text encoder, so that generation is conditioned on chromosome identity.
Trained on the SMU high-performance computing cluster under SLURM.
Evaluation

Visual grids are not evidence, so generation is scored automatically:

Identity classifier. ResNet-18 trained on real crops, ~99% accuracy on held-out spreads.
Validity check. A structural filter that 97% of real chromosomes pass, applied to generated images across multiple seeds rather than a single grid.
Reported metrics. Validity rate, group accuracy (chromosome group), class accuracy (individual chromosome).
Results
	Valid (across seeds)	Group accuracy	Class accuracy
Model 1 — rank-16 LoRA, attention layers	61%	28%	10%
Model 2 — rank-64 LoRA, attention + convolution	89%	50%	19%

Model 1 looked acceptable on a single-seed grid and failed under measurement. It leaked the base model (wood textures, fragments of text), tied background brightness to chromosome size, and learned size from a shortcut in the caption template rather than from chromosome identity.

Model 2 addressed those causes rather than training longer: a single caption template with the group shortcut removed, offset noise, and a higher-rank LoRA extended to the convolution layers. Backgrounds are clean and the base-model leakage is gone.

Current limitations
Banding patterns are not reproduced, so chromosomes within the same group are not yet distinguishable. Anything that depends on band structure is out of scope for the current model.
Class accuracy at 19% is far from usable; group-level structure is learned, individual identity largely is not.
No overlap data has been generated yet, and the downstream segmentation experiment has not been run.
Next steps
Full UNet fine-tuning and a from-scratch baseline, to establish whether the limit is model capacity or the VAE's latent compression.
Compositing overlaps directly, so the segmentation mask is known by construction, using the generative model to make the composite resemble real microscopy rather than to produce the labels.
The downstream test, fixed in advance: a segmentation model trained on generated overlaps must outperform the same model trained on copy-paste overlaps, evaluated on real clusters. Fidelity metrics alone do not establish that synthetic data is fit for purpose.
Repository layout
data/        preprocessing and canvas construction
train/       LoRA fine-tuning scripts
eval/        identity classifier, validity check, metric reporting
samples/     a small set of generated examples
Author

Balakrishnan Arumugam — MSc Computer Science, Southern Methodist University balakrishnanarumugam08@gmail.com · github.com/Krish-008
