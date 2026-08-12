# SwinUNETR (BraTS-GoAT)

Full SwinUNETR pipeline for the BraTS Generalizability Across Tumors (GoAT) task:
supervised training, self-supervised pretraining, region-wise SSL fine-tuning, and a
Docker-packaged inference entrypoint for Synapse submission.

## Layout

```
swinunetr/
├── notebooks/
│   ├── 01_train_supervised.ipynb   # baseline SwinUNETR, 4-class softmax head
│   ├── 02_ssl_pretrain.ipynb       # SSL pretraining (inpainting + rotation + contrastive)
│   └── 03_ssl_finetune.ipynb       # region-wise fine-tuning, 3-region sigmoid head
└── docker/
    ├── Dockerfile                  # pytorch 2.5.1 / cu121 base
    ├── predict.py                  # submission entrypoint (region sigmoid decode)
    ├── requirements.txt            # slim, pinned; torch comes from base image
    ├── build.sh / run.sh / save.sh # build, local test, export helpers
    └── model/                      # place model.pth here (gitignored)
```

## Two head conventions (important)

The pipeline uses two different output heads. Do not mix them.


`predict.py` hard-fails at load time if the checkpoint head does not have 3 channels,
so a softmax checkpoint will not silently run through the submission container.

## Pipeline stages

1. **Supervised baseline** (`train_swinunetr_supervised.ipynb`)


2. **SSL pretraining** (`SSL_pretrain_swinunetr.ipynb`)


3. **Region-wise SSL fine-tuning** (`finetune_swinunetr_ssl_region.ipynb`)
  

## Inference and post-processing

`predict.py` runs, per case:

1. 8-view flip TTA, averaging sigmoid probabilities.
2. 0.5 threshold to binary `[TC, WT, ET]`.
3. Per-region connected-component filtering (min voxels TC=50, WT=100, ET=30).
4. Per-region binary closing (1 iteration, 6-connectivity).
5. Nesting enforced last: `ET subset TC subset WT`.
6. Decode to integer labels and invert back to the input's original image space.

Sliding window: ROI 96^3, batch 4, overlap 0.7, gaussian blending.

## Build and submit

```bash
cd docker
# 1. put the fine-tuned 3-region checkpoint at model/model.pth
./build.sh brats-goat-swinunetr-ssl latest
# 2. local sanity check against a folder of cases
./run.sh /path/to/input /path/to/output
# 3. export for upload
./save.sh brats-goat-swinunetr-ssl latest
```

Input is one subfolder per subject in BraTS naming (`*-t1n/-t1c/-t2w/-t2f.nii.gz`).
Output is one `<subject_id>.nii.gz` label map per subject.
