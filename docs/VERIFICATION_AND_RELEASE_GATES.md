# Evidence and release gates for EGO-Det V2

## Evidence directly cross-checked

- Original `eval_history.jsonl`: 12 full COCO evaluations + 35 `quick500` evaluations. The last full record is epoch 12 / optimizer step 89037.
- Second full-only evaluation record: byte-level equivalent JSON object to the original final full record.
- `ablation_manifest.json`: `source_run=outputs/ego_det_v2/train_mod10b_batch8_long`, `source_checkpoint=epoch_12.pth`, `epoch=12`, no EMA, full AP=0.4526892183243793.
- `all_interventions.csv`: `full` control AP=45.26892183243793 (%), consistent with above.
- `intervention_results.json`: inference-time tests from the same epoch-12 raw weights, **not independently retrained variants**.
- Original `train_log.jsonl`: 7319 records, with one optimizer step rewind during epoch index 3, compatible with resume/replayed logging; train run nonetheless reached 12 full evaluations. Logs alone do not prove checkpoint bytes.

## Unresolved items

1. Verify actual file `epoch_12.pth` exists, its size (manifest: 562663950 bytes), SHA256, schema and `state_dict` keys. The source archive did not include this file.
2. **Current source is not cryptographically tied to the original checkpoint.** Present source has fields `freeze_backbone` and `full_eval_last_epoch_only` absent from the historical manifest, suggesting code may have evolved after the original run. Determine whether these fields alter loaded model structure, and test strict checkpoint loading.
3. Install the dependencies in a clean environment; execute CPU contracts, strict state load, subset inference and full val2017 evaluation (target 45.269 AP).
4. Review license and attribution for `stage1/overlock_official_nofrills.py` and pretrained OverLoCK weights. No license has been assumed or attached.
5. Validate reproducible train settings with actual launcher; the current `EgoDetV2Config` *defaults* are not the historical run (`batch_size=4` vs 8; `grad_accum_steps=4` vs 2; 7 short-side scales vs 5; `efficiency_mode=reference` vs optimized).
6. Audit code/history for secrets, machine paths, personal information and copyrighted third-party assets. The uploaded original ZIP contained an `.env` with live-looking API keys; it is **excluded** from this package. Rotate any real keys and never publish the original archive.
7. Confirm current publisher/author permissions and double-anonymous review restrictions, if applicable, before publicizing an official implementation.

## Verification commands

```bash
python -m compileall -q ego_det_v2 stage1 ego_joint_object_slot
python -m ego_det_v2.launch.launch_contracts_cpu --fake-backbone
python tools/check_checkpoint_metadata.py checkpoints/epoch_12.pth
bash scripts/evaluate_paper_epoch12.sh
```

Local validation: `compileall` passed; training/eval CLI help passed; ablation and inference-intervention tests passed. CPU fake-backbone contracts passed 37 checks then stopped when `pycocotools` was missing. GPU, checkpoint strict loading and full COCO AP have not been performed.

## Historical method configuration

- 300 slots, 3 refinement rounds, 48 local proposals per slot.
- Proposal bank: 1000 train / 1500 eval; FPN P3–P6.
- Diversity seeding, semantic global attention and regional retrieval, slot competition.
- Boundary readout with dustbin gating and bounded residual correction.
- AdamW: base LR 1e-4, backbone LR 1e-5, stem LR 1e-6; 12 epochs, drop at 11; batch 8, accumulation 2; bf16 AMP; seed 3407.
