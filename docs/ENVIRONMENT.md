# EGO-Det environment

**Status:** PyTorch/CUDA and the repository's direct runtime dependencies were installed successfully in an independent `egodet` Conda environment on WSL2 with an RTX 2080 Ti. The raw `epoch_12.pth` loaded strictly and CUDA inference succeeded. COCO AP reproduction is still pending.

## Fresh installation

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate egodet
python -m pip install torch==2.1.2 torchvision==0.16.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pip check
```

`pyproject.toml` defines packaging metadata and reuses `requirements.txt` as the single runtime dependency list. The last command is an editable installation; it does **not** replace the explicit CUDA PyTorch installation.

Avoid mixing the independent `egodet` environment with the older `detector` Conda environment. The latter contained NumPy 2.x and mixed CUDA runtimes and produced a PyTorch/NumPy binary-interface warning. The working `egodet` runtime uses NumPy 1.26.4 and PyTorch 2.1.2+cu118. No MMDetection, MMCV, `nvcc`, or custom GEMM build was needed for initial inference.

## Smoke commands

```bash
python -m egodet.infer --help
python -m ego_det_v2.launch.launch_eval_ego_det_v2 --help
python tools/check_source_layout.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

An RTX 2080 Ti does not natively support BF16. The evaluation path falls back to FP16 when BF16 is unavailable. This may produce small differences from the original BF16 training/evaluation environment and does not guarantee paper-scale training fits within 11 GiB.

## Still to verify before public release

- Run validation on an actual COCO image subset using the official evaluator.
- Run full COCO val2017 and compare with the *historical* AP 45.269%.
- Confirm all third-party notices, rights, and author/review permissions.

Do not change the recorded paper configuration to hide device-specific limitations.
