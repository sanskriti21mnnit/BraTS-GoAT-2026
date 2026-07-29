# BraTS-GoAT

Segmentation pipelines for the BraTS Generalizability Across Tumors (GoAT) task.
Each model lives in its own folder with the same structure: training notebooks
(supervised, SSL pretraining, region-wise SSL fine-tuning) plus a Docker-packaged
inference entrypoint for Synapse submission.

## Structure

```
brats-goat/
├── README.md
├── LICENSE
├── requirements-dev.txt        # local training / notebook environment
├── .gitignore
├── models/
│   ├── swinunetr/              # SwinUNETR pipeline (implemented)
│   │   ├── notebooks/          # 01 supervised, 02 SSL pretrain, 03 SSL finetune
│   │   └── docker/             # Dockerfile, predict.py, requirements, helpers, model/
│   └── unet/                   # 3D U-Net pipeline (blank mirror, to be filled in)
│       ├── notebooks/
│       └── docker/
└── scripts/
    └── init_git.sh             # one-shot git init + first commit
```

## Task convention

- Inputs: 4 modalities per subject, `[t1n, t1c, t2w, t2f]`, BraTS naming.
- Labels: `0=BG, 1=NCR, 2=ED, 3=ET`.
- Evaluation regions: WT `{1,2,3}`, TC `{1,3}`, ET `{3}`.

Two head conventions are used across the pipeline. The supervised baseline uses a
4-class softmax head. The SSL fine-tuned submission model uses a 3-region sigmoid
head `[TC, WT, ET]` decoded to the integer label map. See each model's README.

## Quick start

```bash
# local dev environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# work through the SwinUNETR notebooks in order
#   models/swinunetr/notebooks/01_train_supervised.ipynb
#   models/swinunetr/notebooks/02_ssl_pretrain.ipynb
#   models/swinunetr/notebooks/03_ssl_finetune.ipynb

# build and submit (see models/swinunetr/README.md)
cd models/swinunetr/docker
./build.sh && ./run.sh /path/to/input /path/to/output && ./save.sh
```

## Data and weights

Data and `.pth` checkpoints are gitignored and must never be committed. Keep patient
scans (including any private hospital cohort) out of the repo entirely, since deleting
a committed file does not remove it from git history. Share final checkpoints via a
GitHub Release or your Synapse project.

## Models

| Model     | Status      | Notes                                              |
|-----------|-------------|----------------------------------------------------|
| SwinUNETR | Implemented | supervised + SSL pretrain + region-wise fine-tune  |
| 3D U-Net  | Placeholder | blank mirror, same structure, to be implemented    |
