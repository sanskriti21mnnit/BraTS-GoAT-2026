# Model weights

Place the trained SwinUNETR-SSL checkpoint here as `model.pth`.

This must be the **3-region sigmoid** head (`OUT_CHANNELS = 3`, regions `[TC, WT, ET]`)
produced by `../../notebooks/03_ssl_finetune.ipynb`. The Dockerfile copies
`model/model.pth` to `/opt/model/model.pth` and `predict.py` loads it from there.

Weights are gitignored. Do not commit `.pth` files. Share final checkpoints via a
GitHub Release or your Synapse project instead.
