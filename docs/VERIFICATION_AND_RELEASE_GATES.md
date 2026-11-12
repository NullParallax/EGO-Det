# Evidence and release gates for EGO-Det V2

## Historical evidence (not a new COCO evaluation)

- The recorded epoch-12 final model used the raw (`model`), not EMA, state.
- Historical full COCO val2017: AP 45.2689%, AP50 63.3310%, AP75 49.4821%.
- The published run used 12 epochs, batch size 8, gradient accumulation 2, 300 object slots, and 48 evidence proposals per slot.
- `configs/paper_epoch12_recorded.json` preserves the recorded training configuration for reference. Current class defaults differ in some training fields.

## Runtime checks completed on WSL2 / RTX 2080 Ti

- Dedicated Python 3.11 `egodet` environment with PyTorch 2.1.2+cu118 and NumPy 1.26.4 installed.
- Final trusted `epoch_12.pth` raw model loaded with `strict=True` using the current model implementation.
- GPU forward ran without NaN/Inf; output shapes were boxes `[1,300,4]`, logits `[1,300,80]`, scores `[1,300,80]`.
- A real single-image demonstration completed successfully; this is not an AP measurement.
- Source-only compatibility tests and Python compilation have passed in the staging workflow.

## Remaining before a public academic release

1. Run a small COCO validation subset, then the complete val2017 dataset with the repository evaluator; report the *newly measured* metrics separately from historical records.
2. Confirm the reproducible train command and actual resource requirements; GPU hardware differs from the original training environment.
3. Review the retained third-party OverLoCK source, pretrained-weight redistribution permissions, and upstream license notices (`docs/THIRD_PARTY.md`).
4. Review the entire source and Git history for private credentials and machine paths before switching the repository to Public.
5. Obtain applicable author approval and check double-anonymous submission restrictions.

## Recommended evaluation command

```bash
bash scripts/evaluate_paper_epoch12.sh
```

On RTX 2080 Ti, BF16 is unavailable and the evaluator selects FP16 for CUDA AMP. Report this difference when comparing with recorded BF16 results.
