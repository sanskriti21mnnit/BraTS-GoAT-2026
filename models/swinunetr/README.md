# SwinUNETR (BraTS-GoAT)

Full SwinUNETR pipeline for the BraTS Generalizability Across Tumors (GoAT) task:
supervised training, self-supervised pretraining, region-wise SSL fine-tuning, and a
Docker-packaged inference entrypoint for Synapse submission.

## Layout

```
swinunetr/
│   ├── train_swinunetr_supervised.ipynb   # baseline SwinUNETR,
│   ├── SSL_pretrain_swinunetr.ipynb       # SSL pretraining (inpainting + rotation + contrastive)
│   └── finetune_swinunetr_ssl_region.ipynb       # region-wise fine-tuning, 3-region sigmoid head
└── docker/
    ├── Dockerfile                  # pytorch 2.5.1 / cu121 base
    ├── predict.py                  # submission entrypoint (region sigmoid decode)
    ├── requirements.txt            # slim, pinned; torch comes from base image
    ├── build.sh / run.sh / save.sh # build, local test, export helpers
    └── model/                      # place model.pth here (gitignored)
```



**BraTS-GoAT Challenge: Tumor Segmentation**

This repository contains the training and inference pipelines developed for the BraTS-GoAT Challenge for automated brain tumor segmentation from multimodal MRI scans. The framework includes two complementary deep learning pipelines: a fully supervised learning pipeline and a self-supervised learning (SSL) pipeline.

1. **Overview**

The objective of this work is to develop robust 3D deep learning models for brain tumor segmentation using the multimodal MRI sequences provided by the BraTS-GoAT dataset.

The proposed framework investigates two different training strategies:

**Supervised Learning Pipeline** – the segmentation model is trained directly using labeled MRI volumes and corresponding tumor segmentation masks.
**Self-Supervised Learning Pipeline** – an encoder is first pretrained using unlabeled MRI data through a self-supervised learning objective. The pretrained encoder is subsequently transferred to the downstream tumor segmentation task.


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
