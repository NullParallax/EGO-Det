"""Read-only local checkpoint identity inspection; never publishes tensors."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--hash', action='store_true', help='calculate SHA256 of entire file')
    args=parser.parse_args()
    path=args.checkpoint.expanduser().resolve()
    report={'checkpoint_name':path.name,'size_bytes':path.stat().st_size}
    if args.hash:
        sha=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):
                sha.update(block)
        report['sha256']=sha.hexdigest()
    print('Reading trusted local checkpoint on CPU...')
    ckpt=torch.load(str(path),map_location='cpu',weights_only=False)
    if not isinstance(ckpt,dict):
        raise TypeError(f'Unexpected checkpoint type: {type(ckpt)}')
    report['checkpoint_keys']=list(ckpt.keys())
    for k in ['epoch','global_step','global_optimizer_step','micro_step']:
        if k in ckpt and isinstance(ckpt[k],(str,int,float)):
            report[k]=ckpt[k]
    if 'ema' in ckpt:
        report['ema_present']=ckpt['ema'] is not None
    model=ckpt.get('model')
    if isinstance(model,dict):
        report['model_tensor_count']=len(model)
        report['first_20_model_keys']=list(model)[:20]
    cfg=ckpt.get('config')
    if isinstance(cfg,dict):
        for k in ['batch_size','grad_accum_steps','epochs','train_short_sides','efficiency_mode','num_slots','bank_topk_train','bank_topk_eval','evidence_per_slot','use_ema','seed','amp_dtype','freeze_backbone']:
            if k in cfg:report.setdefault('config',{})[k]=cfg[k]
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
