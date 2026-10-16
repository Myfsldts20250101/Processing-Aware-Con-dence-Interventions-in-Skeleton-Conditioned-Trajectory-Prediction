"""Small O/F/N_J/N inference check; never produces a formal NJ_DONE marker."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, os, sys, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'diagnostics')]
sys.dont_write_bytecode = True

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default=str(ROOT / 'local_runs/smoke'))
    p.add_argument('--max-targets', type=int, default=8)
    p.add_argument('--train-seeds', default='42')
    args = p.parse_args()
    out = Path(args.out_dir).resolve()
    out.relative_to(ROOT / 'local_runs')
    out.mkdir(parents=True, exist_ok=True)
    report = {'status':'STARTED', 'scope':'small fixed-roster four-condition check; not full evaluation', 'checkpoints':[], 'inference_executed':False}
    def save():
        (out/'SMOKE_REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    missing = [n for n in ('torch','numpy','yaml') if importlib.util.find_spec(n) is None]
    if missing:
        report.update(status='BLOCKED', missing_dependencies=missing)
        save(); print(json.dumps(report)); return 2
    try:
        import numpy as np
        import torch
        import yaml
        from data.jrdb_dataset import build_dataloader
        from models.full_model import build_model
        from eval_u1f_cell import assign_conf_F, predict_locked
        from e123_common import batch_window_keys
        from e1_rev1_common import load_donor_map, lookup_donor_in_batch, ade_fde_vs_gt
        os.chdir(ROOT)
        os.environ['CTR_GCN_TRAJ_MOTION_CACHE_ROOT'] = str(out/'motion_filter_cache')
        torch.set_num_threads(2)
        torch.manual_seed(0)
        cfg=yaml.safe_load((ROOT/'configs/jrdb_v2_decoupled_z.yaml').read_text(encoding='utf-8'))
        cfg['data']['data_dir']=str(ROOT/'data/jrdb_processed')
        cfg['data']['num_workers']=0
        cfg['data'].setdefault('motion_filter',{})['enabled']=True
        # Preserve the original batch size and donor-in-batch lookup.
        roster=set(json.loads((ROOT/'protocols/diagnostic_BV/diagnostic_set_BV.json').read_text(encoding='utf-8'))['target_keys'])
        meta,pairs=load_donor_map(ROOT/'protocols/donor_maps/donor_pairs_B_d42.json')
        assert int(meta['donor_assign_seed'])==42
        loader=build_dataloader(cfg,split='test')
        chosen=None
        for batch in loader:
            keys=batch_window_keys(batch); targets=[]; donors=[]
            for i,key in enumerate(keys):
                if key not in roster: continue
                j,status,_=lookup_donor_in_batch(pairs,key,keys)
                if status=='ok': targets.append(i);donors.append(j)
                if len(targets)>=args.max_targets:break
            if targets:
                chosen=(batch,keys,targets,donors);break
        assert chosen is not None, 'No frozen target/donor pairs found in loader batches'
        batch,keys,targets,donors=chosen
        report.update(python=sys.version,torch=torch.__version__,device='cpu',donor_seed=42,target_keys=[keys[i] for i in targets],donor_keys=[keys[j] for j in donors])
        for seed in [int(x) for x in args.train_seeds.split(',')]:
            paths=list((ROOT/'checkpoints').glob(f'jrdb_v2_decoupled_z_s{seed}_*/best.pt'))
            assert len(paths)==1, f'Checkpoint count for seed {seed}: {len(paths)}'
            checkpoint=paths[0]
            h=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            registry=(ROOT/'checkpoints/REGISTRY.md').read_text(encoding='utf-8')
            assert any(checkpoint.parent.name in row and h in row for row in registry.splitlines())
            # Trusted author-supplied checkpoints include saved configuration.
            blob=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model_cfg=copy.deepcopy(blob.get('config',cfg)); model_cfg['data']=cfg['data']
            model=build_model(model_cfg,verbose=False)
            model.load_state_dict(blob['model_state_dict'],strict=True)
            model.eval()
            obs=batch['obs_traj'];sk=batch['skeleton_kp'];sc=batch['skeleton_conf']
            scd=sc.clone()
            for i,j in zip(targets,donors):scd[i]=sc[j]
            cj,jt=assign_conf_F(sc,scd)
            with torch.inference_mode():
                xo,mo=model.skeleton_preprocessor(sk.clone(),sc.clone())
                xf=xo.clone();xf[:,2]=cj
                o=model.predict(obs,sk.clone(),sc.clone(),num_samples=1,predict_mode='most_likely')
                f=predict_locked(model,obs,xf,mo)
                nj=model.predict(obs,sk.clone(),cj.clone(),num_samples=1,predict_mode='most_likely')
                n=model.predict(obs,sk.clone(),scd.clone(),num_samples=1,predict_mode='most_likely')
                replay=predict_locked(model,obs,xo,mo)
                repeat=model.predict(obs,sk.clone(),sc.clone(),num_samples=1,predict_mode='most_likely')
            report['inference_executed']=True
            preds={'O':o,'F':f,'NJ':nj,'N':n}
            assert all(tuple(v.shape)==(obs.shape[0],1,12,2) for v in preds.values())
            assert all(bool(torch.isfinite(v).all()) for v in preds.values())
            repeat_max=float((o-repeat).abs().max());replay_max=float((o-replay).abs().max())
            assert repeat_max<=1e-6 and replay_max<=1e-6
            assert torch.equal(xf[:,:2],xo[:,:2])
            assert torch.equal(cj[~jt],sc[~jt])
            ades={k:ade_fde_vs_gt(v,batch['pred_traj'])[0][targets].cpu().numpy() for k,v in preds.items()}
            np.savez_compressed(out/f's{seed}_d42_predictions.npz',keys=np.array(report['target_keys'],dtype=str),gt=batch['pred_traj'][targets].numpy(),**{f'pred_{k}':v[targets,0].numpy() for k,v in preds.items()})
            report['checkpoints'].append({'seed':seed,'sha256':h,'strict_state_dict':True,'n_targets':len(targets),'n_forward_batch':obs.shape[0],'repeat_max_abs':repeat_max,'locked_replay_max_abs':replay_max,'mean_ADE':{k:float(v.mean()) for k,v in ades.items()}})
            save();del model,blob
        report['status']='PASS';save();print(json.dumps(report,ensure_ascii=False));return 0
    except Exception:
        report.update(status='FAIL',error=traceback.format_exc());save();raise

if __name__=='__main__':
    raise SystemExit(main())
