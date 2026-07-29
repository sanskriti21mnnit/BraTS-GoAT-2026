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

- `01_train_supervised.ipynb` uses a **4-class softmax** head (`OUT_CHANNELS = 4`,
  labels BG=0, NCR=1, ED=2, ET=3). This is the from-scratch baseline.
- `03_ssl_finetune.ipynb` and `docker/predict.py` use a **3-region sigmoid** head
  (`OUT_CHANNELS = 3`, regions `[TC, WT, ET]`), decoded to the integer label map
  1=NCR, 2=ED, 3=ET. The Docker image expects this checkpoint.

`predict.py` hard-fails at load time if the checkpoint head does not have 3 channels,
so a softmax checkpoint will not silently run through the submission container.

## Pipeline stages

1. **Supervised baseline** (`01_train_supervised.ipynb`)
   SwinUNETR `feature_size=48` (about 62M params), 96^3 patches, AMP, cosine schedule,
   85/15 train/val split, early stopping. Produces the baseline segmentation model.

2. **SSL pretraining** (`02_ssl_pretrain.ipynb`)
   Self-supervised pretraining on unlabeled cases using three proxy tasks: masked
   inpainting (mask ratio 0.75), rotation prediction, and a contrastive objective
   (low weight to avoid representation collapse). Can start from random init or warm
   start from the supervised encoder. Exports the pretrained encoder weights.

3. **Region-wise SSL fine-tuning** (`03_ssl_finetune.ipynb`)
   Loads the SSL encoder and fine-tunes with a 3-region sigmoid head. Includes
   class-aware cropping (ET oversampled), 8-flip TTA, gaussian sliding-window, and the
   post-processing used at submission. Ablation switches (encoder LR, warmup, weight
   decay on norm/bias, augmentation level, EMA) are grouped in the config cell so one
   variable changes per run.

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
