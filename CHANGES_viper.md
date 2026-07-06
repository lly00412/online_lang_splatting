# Viper setup — all changes (reference for re-applying on another branch)

This records every fix made to get `online_lang_splatting` running on the **viper** server
(headless Linux, conda env `LangGS`, torch 1.13, CUDA 11.7, Python 3.8).
Use it to re-apply the changes after switching branches.

---

## 1. Environment / setup fixes (not in the repo — machine-level)

These are not code edits but are needed for a clean run.

### 1a. Conda ran out of disk (`NoSpaceLeftError`)
Root disk `/` was 100% full. Point conda envs/pkgs at the big disk and clean cache:
```bash
conda clean --all
# ~/.condarc:
#   envs_dirs:  [/mnt/Data2/liyan/conda/envs]
#   pkgs_dirs:  [/mnt/Data2/liyan/conda/pkgs]
```

### 1b. torch import crash — cuBLAS mismatch
Error: `libcublas.so.11: undefined symbol: cublasLt_for_cublas_HSS`.
Cause: `LD_LIBRARY_PATH` forced the **system** `/usr/local/cuda-11.7/lib64` libs to
shadow the conda env's own libs. Fix — comment out this line in `~/.bashrc`:
```bash
# line ~138 in ~/.bashrc — COMMENTED OUT:
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```
`nvcc` compiling uses `PATH`, not `LD_LIBRARY_PATH`, so this is safe for building.

### 1c. Lightning package conflict
The unified `lightning` package eagerly imports `lightning.app` -> `lightning_cloud`,
which had a version mismatch (`cannot import name 'AppinstancesIdBody'`).
Fix — use `pytorch_lightning` instead and remove the unified package:
```bash
pip uninstall -y lightning lightning_cloud
# keep: pytorch_lightning==1.9.0  (already installed)
```

---

## 2. PyTorch Lightning import changes  (`lightning.pytorch` -> `pytorch_lightning`)

The code imported the unified `lightning` package; switched all imports to the
already-installed `pytorch_lightning` (API-identical at 1.9.0).

### File: `language/autoencoder/model.py`  (line 4)
```diff
- import lightning.pytorch as pl
+ import pytorch_lightning as pl
```

### File: `language/autoencoder/train_encoder_light.py`  (lines 4, 7, 8, 9, 10)
```diff
- import lightning.pytorch as pl
+ import pytorch_lightning as pl
- from lightning.pytorch.strategies.ddp import DDPStrategy
+ from pytorch_lightning.strategies.ddp import DDPStrategy
- from lightning.pytorch import loggers as pl_loggers
+ from pytorch_lightning import loggers as pl_loggers
- from lightning.pytorch.callbacks import ModelCheckpoint
+ from pytorch_lightning.callbacks import ModelCheckpoint
- import lightning as L
+ import pytorch_lightning as L
```

One-shot sed to re-apply both files:
```bash
sed -i \
  -e 's/import lightning\.pytorch as pl/import pytorch_lightning as pl/' \
  -e 's/from lightning\.pytorch/from pytorch_lightning/g' \
  -e 's/^import lightning as L/import pytorch_lightning as L/' \
  language/autoencoder/model.py \
  language/autoencoder/train_encoder_light.py
```

---

## 3. Language model loader fix  (`language/load_lang_model.py`)

**Problem:** `load_lang_model(model_path=...)` did `torch.load(model_path)` and returned it
directly. When `model_path` is a state-dict / detectron2 checkpoint (e.g. `sed_model_large.pth`)
that returns a `dict`, so `model(inputs)` later failed with `'dict' object is not callable`.

**Fix:** detect a dict vs. a whole pickled model. If it's a checkpoint, build the architecture
from config and load the weights into it.

Function `load_lang_model` now reads:
```python
def load_lang_model(model_path=None):
    if model_path is not None:
        loaded = torch.load(model_path, map_location="cuda:0")
        if not isinstance(loaded, dict):
            # already a full model object
            model = loaded
            model.eval()
            return model
        # loaded is a checkpoint dict -> build model and load the weights into it
        args = get_parser().parse_args([])          # use default config, ignore caller's argv
        cfg = setup_cfg(args)
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(model_path)
    else:
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        model = build_model(cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        model_state_path = "seg_clip_model_l.pth"
        torch.save(model, model_state_path)
    return model
```

---

## 4. SLAM backend fixes  (`utils/slam_backend.py`)

### 4a. Add `os` import (was missing)  — near line 17
```diff
+ import os
  import random
  import time
```

### 4b. Headless guard in `visualize_similarity`  — near line 230
Server has no X display; unconditional `cv2.imshow` crashed with
`qt.qpa.xcb: could not connect to display`. Skip the GUI when `$DISPLAY` is unset.
```python
    def visualize_similarity(self, recon_online, recon_coco):
        """..."""
        # Skip GUI visualization on headless machines (e.g. servers over SSH with no X display)
        if not os.environ.get("DISPLAY"):
            return
        # ... rest unchanged (perform_similarity, cv2.applyColorMap, cv2.imshow, cv2.waitKey)
```
(With SSH X-forwarding active, `$DISPLAY` is set, so the windows still show.)

### 4c. Create output dir before saving online autoencoder  — near line 897
`torch.save` failed with `Parent directory ... does not exist` because the
online checkpoint path pointed at a non-existent dir. Auto-create it.
```diff
  if not self.is_single_stage:
-     # Save online autoencoder weights
-     torch.save(
-         self.online_auto.state_dict(),
-         self.config["language"]["online_ckpt_path"]
-     )
+     # Save online autoencoder weights (ensure the target dir exists)
+     online_ckpt_path = self.config["language"]["online_ckpt_path"]
+     os.makedirs(os.path.dirname(online_ckpt_path), exist_ok=True)
+     torch.save(
+         self.online_auto.state_dict(),
+         online_ckpt_path
+     )
      Log("Saved online autoencoder weights")
```

---

## 5. Config parameter changes  (`configs/rgbd/replicav2/`)

Original configs hardcoded the author's disk paths (`/media/saimouli/...`, `/media/Replica/...`).
Repoint `online_ckpt_path` to your own disk.

### DONE — `room0.yaml`  (line 22)
```diff
- online_ckpt_path: "/media/saimouli/Data6T/Replica/omni_data_result/online_15_room0.pth"
+ online_ckpt_path: "/mnt/Data2/liyan/online_lang_splatting/results/replica/room0/omni_data_result/online_15_room0.pth"
```

### STILL TODO — other scenes (only if you run them)
These still point at author paths and need the same treatment:

| File          | line | current value |
|---------------|------|---------------|
| `room1.yaml`  | 22   | `/media/Replica/omni_data_result/online_15_room0.pth` |
| `room2.yaml`  | 22   | `/media/Replica/omni_data_result/online_15_room0.pth` |
| `office0.yaml`| 20   | `/media/Replica/omni_data_result/online_15_office0.pth` |
| `office1.yaml`| 20   | `/media/Replica/omni_data_result/online_15_office1.pth` |
| `office2.yaml`| 20   | `/media/Replica/omni_data_result/online_15_office2.pth` |
| `office3.yaml`| 20   | `/media/Replica/omni_data_result/online_15_office3.pth` |
| `office4.yaml`| 20   | `/media/Replica/omni_data_result/online_15_office4.pth` |

Note: the `makedirs` fix (4c) means these no longer *crash* on a missing dir, but they
would still try to write under `/media/...` (likely not writable), so repoint them anyway.

Other relevant params already set in `room0.yaml`:
- `Dataset.dataset_path: /mnt/Data4/slam_datasets/vmap/room_0/imap/00/`
- `Results.use_gui: False`
- `language.online_ckpt_path` (changed above)
- SED model weights (`convnextL_768.yaml`): `MODEL.WEIGHTS: /mnt/Data2/liyan/online_lang_splatting/data/OnlineLanguageSplatting/sed_model_large.pth`

---

## 6. `.gitignore` additions (Windows local copy)

Added to keep run outputs / datasets / model blobs out of git:
```
/results/
/data/
/media/
*.pth
*.pkl
*.ply
*.ckpt
*.npy
*.Xauthority
```

---

## Files changed (summary)
- `language/autoencoder/model.py`            — lightning import
- `language/autoencoder/train_encoder_light.py` — lightning imports
- `language/load_lang_model.py`              — dict-vs-model loader fix
- `utils/slam_backend.py`                    — `import os`, headless guard, makedirs before save
- `configs/rgbd/replicav2/room0.yaml`        — online_ckpt_path -> your disk
- `.gitignore`                               — ignore results/data/model blobs
- (machine-level) `~/.bashrc`, `~/.condarc`  — LD_LIBRARY_PATH, conda dirs
