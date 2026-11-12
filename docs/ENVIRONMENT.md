# EGO-Det V2: minimal paper-compatible environment

Status: **proposed; not yet installed/tested in the user's WSL**. This configuration is intentionally minimal and does not claim to be an exact snapshot of the original server.

## 1. Why not use the old environment.yml?

The exported server `detector` environment contains packages unrelated to the source imports (e.g. `mmdet`, `mmcv`, `mmengine`, `wandb`, `openai`, `opencv-python`) and mixed Conda `libtorch` plus pip CUDA 11/12 runtime packages. The formal V2 code directly imports `torch`, `torchvision`, `numpy`, `scipy`, `PIL`, `pycocotools`, `timm`, `einops`, and `tqdm`. Python standard-library modules do not need to be listed. The OverLoCK optional `natten` and implicit-GEMM imports have fallback paths in the source and are not required for first boot.

## 2. Install (later, only after Git baseline is recorded)

Run in WSL within the repository root:

```bash
conda env create -f environment.yml
conda activate egodet

# Official PyTorch 2.1.2 CUDA 11.8 wheels; installs associated runtime libraries.
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Runtime dependencies (torch is intentionally NOT repeated here).
python -m pip install -r requirements.txt
python -m pip check
```

There is no need to install a system-wide CUDA Toolkit or `nvcc` just to load the prebuilt PyTorch wheel, unless a later feature requires compiling custom CUDA extensions. Do not install `torchaudio`, MMDetection, MMCV, OpenMIM, W&B, or any NVIDIA CUDA *conda* meta-packages for this V2 baseline without a demonstrated import/runtime requirement.

Development-only checks can be enabled separately:

```bash
python -m pip install -r requirements-dev.txt
```

## 3. Version and GPU validation (not executed yet)

```bash
python - <<'PY_CHECK'
import torch, torchvision, numpy, scipy, timm, einops, pycocotools
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('BF16 supported:', torch.cuda.is_bf16_supported())
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('timm:', timm.__version__)
print('einops:', einops.__version__)
PY_CHECK

python -m compileall -q ego_det_v2 ego_joint_object_slot stage1
python -m ego_det_v2.launch.launch_eval_ego_det_v2 --help
```

## 4. WSL RTX 2080 Ti limitations

The observed WSL GPU is an RTX 2080 Ti with 11 GiB memory. It supports the CUDA 11.8 PyTorch binary under a sufficiently recent NVIDIA driver but **does not natively support BF16**. The V2 training/evaluator code checks `torch.cuda.is_bf16_supported()` and selects FP16 otherwise. Therefore, local evaluation may not exactly match the paper's original BF16 AP, and paper-scale batch 8 training may exceed 11 GiB. Do **not** edit the historical `configs/paper_epoch12_recorded.json` to fit the local GPU; distinguish hardware-local settings from official reproduction settings.

Before treating the installation as validated, strictly load the epoch-12 checkpoint, run a small COCO subset, then run full val2017 and compare with the historical 45.2689 AP.

## 5. Not included yet

A cross-version lockfile with wheel hashes, official test/CI matrix, and CPU-only installation are not validated. Keep `environment.yml` and `requirements.txt` as **installation candidates** until the actual smoke checks complete.
