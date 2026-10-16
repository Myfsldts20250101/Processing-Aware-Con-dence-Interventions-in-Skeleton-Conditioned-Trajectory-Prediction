#!/usr/bin/env python3
"""Reproduce current core statistics using only this package and Python's standard library."""
from pathlib import Path
from collections import defaultdict
import csv, gzip, json, math, statistics, hashlib, argparse, sys

HERE=Path(__file__).resolve().parent

def mean(a): return statistics.fmean(a)
def summary(seed_values):
    v=[seed_values[s] for s in sorted(seed_values)]
    return {'mean':mean(v),'sample_sd':statistics.stdev(v),'positive_seeds':sum(x>0 for x in v),'negative_seeds':sum(x<0 for x in v),'zero_seeds':sum(x==0 for x in v),'seed_values':{str(s):seed_values[s] for s in sorted(seed_values)}}
def collapse(cell_values):
    seeds=defaultdict(list)
    for (s,d),v in sorted(cell_values.items()): seeds[s].append(v)
    assert sorted(seeds)==list(range(42,50)) and all(len(v)==5 for v in seeds.values())
    return summary({s:mean(v) for s,v in seeds.items()})
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def load_rows(path):
    with gzip.open(path,'rt',encoding='utf-8',newline='') as f: yield from csv.DictReader(f)
def calc():
    manifest=json.loads((HERE/'derived_input_hashes.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path=HERE/item['path']
        assert path.is_file() and path.stat().st_size==item['bytes'],('file size',item['path'])
        assert sha(path)==item['sha256'],('SHA-256 mismatch',item['path'])
    roster={r['target_key']:r['scene_id'] for r in load_rows(HERE/'data/frozen_roster.csv.gz')}
    donor_maps={d:{r['target_key']:r['donor_key'] for r in load_rows(HERE/f'data/donor_map_d{d}.csv.gz')} for d in range(42,47)}
    assert len(roster)==15160 and all(set(m)==set(roster) for m in donor_maps.values())
    allmetrics=defaultdict(dict); groups=defaultdict(lambda:defaultdict(dict)); groupinfo=defaultdict(dict)
    current=None; rows=[]; count=0; crossgap=0.0; identitymax=0.0
    def finish(cell,rr):
        assert len(rr)==15160
        pairset={(r['target_key'],r['donor_key']) for r in rr}
        assert len(pairset)==15160 and {r['target_key'] for r in rr}==set(roster)
        for col in ['ade_O','ade_F','ade_NJ','ade_N','R_policy','R_support','delta_ade_F','delta_ade_NJ','delta_ade_N']:
            allmetrics[col][cell]=mean([float(r[col]) for r in rr])
        for name,col in [('policy','delta_policy'),('support','delta_support')]:
            v=[float(r[col]) for r in rr]
            vals={'H':mean([max(x,0.) for x in v]),'I':mean([max(-x,0.) for x in v]),'delta':mean(v),'p_positive':sum(x>0 for x in v)/len(v),'p_negative':sum(x<0 for x in v)/len(v),'p_zero':sum(x==0 for x in v)/len(v)}
            assert abs(vals['H']-vals['I']-vals['delta'])<1e-12
            for k,z in vals.items():allmetrics[name+'.'+k][cell]=z
        for cmp,base,metric in [('N_minus_O','n_frame_O','delta_ade_N'),('N_minus_NJ','n_frame_NJ','delta_support')]:
            for label,sign in [('unchanged',0),('increased',1),('decreased',-1)]:
                subset=[r for r in rr if ((int(r['n_frame_N_u1_label'])>int(r[base]))-(int(r['n_frame_N_u1_label'])<int(r[base])))==sign]
                name=cmp+'.'+label
                if not subset:
                    groupinfo[name][cell]={'n':0,'scenes':0}
                    continue
                byscene=defaultdict(list)
                for r in subset:byscene[r['scene_id']].append(float(r[metric]))
                groups[name]['window_weighted'][cell]=mean([float(r[metric]) for r in subset])
                groups[name]['scene_equal'][cell]=mean([mean(v) for v in byscene.values()])
                groupinfo[name][cell]={'n':len(subset),'scenes':len(byscene)}
    for r in load_rows(HERE/'data/core_windows.csv.gz'):
        cell=(int(r['train_seed']),int(r['donor_map']))
        assert 42<=cell[0]<=49 and 42<=cell[1]<=46
        assert r['scene_id']==roster[r['target_key']] and donor_maps[cell[1]][r['target_key']]==r['donor_key']
        assert int(r['include_main'])==int(r['u1_include_main'])==1
        assert int(r['fail_exec'])==int(r['nonfinite'])==0 and int(r['assign_ok'])==int(r['mono_ok'])==1
        assert int(r['n_frame_O'])==int(r['u1_n_frame_O_audit_only'])
        assert int(r['n_frame_NJ'])<=int(r['n_frame_O'])
        assert int(r['n_frame_N_u1_label'])>=int(r['n_frame_NJ'])
        for k,v in r.items():
            if k not in ['scene_id','target_key','donor_key']: assert math.isfinite(float(v))
        crossgap=max(crossgap,abs(float(r['delta_ade_N'])-float(r['u1_delta_ade_N_audit_only'])))
        identitymax=max(identitymax,abs(float(r['delta_ade_N'])-float(r['delta_ade_F'])-float(r['delta_policy'])-float(r['delta_support'])))
        assert abs(float(r['ade_N'])-float(r['ade_NJ'])-float(r['delta_support']))<1e-9
        assert abs(float(r['ade_NJ'])-float(r['ade_F'])-float(r['delta_policy']))<1e-9
        if current is None:current=cell
        if cell!=current:finish(current,rows);rows=[];current=cell
        rows.append(r);count+=1
    if rows:finish(current,rows)
    assert count==606400 and len(allmetrics['ade_O'])==40
    assert identitymax<1e-6
    report={'schema_version':'1.0','scope':'Same-run NJ O/F/NJ/N core statistics; U1 contributes N frame-count labels only.','aggregation':'Within each seed/donor cell average windows, or average scene means within the group over scenes where that group occurs; average five donor values within seed; summarize eight seed means with sample SD (ddof=1).','sign_rule':'Strict > 0 / < 0 / == 0 on stored differences; no epsilon.','audit':{'cells':40,'rows':count,'unique_target_windows':len(roster),'unique_scenes':len(set(roster.values())),'max_delta_N_crossrun_abs':crossgap,'max_additive_identity_residual':identitymax,'clean_frame_count_mismatches':0},'core':{k:collapse(v) for k,v in sorted(allmetrics.items())},'frame_groups':{}}
    for name,details in sorted(groupinfo.items()):
        ns=[v['n'] for v in details.values()];ss=[v['scenes'] for v in details.values()]
        item={'mean_windows_per_cell':mean(ns),'window_count_range':[min(ns),max(ns)],'scene_count_range':[min(ss),max(ss)],'cell_counts':{f's{s}/d{d}':v for (s,d),v in sorted(details.items())}}
        if name in groups:
            item.update({k:collapse(v) for k,v in groups[name].items()})
        else:item['status']='No windows in any cell; effect undefined.'
        report['frame_groups'][name]=item
    return report

def compare(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),(path,'keys differ')
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),(path,'length differs')
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(path,a,b)
    else:assert a==b,(path,a,b)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--write-expected',action='store_true');p.add_argument('--output',default='reproduced_results.json');args=p.parse_args()
    out=(HERE/args.output).resolve()
    assert out.parent==HERE.resolve(),'Output must stay in this package directory.'
    result=calc()
    expected=HERE/'expected_results.json'
    if args.write_expected: expected.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    else: compare(result,json.loads(expected.read_text(encoding='utf-8')))
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':'PASS','cells':result['audit']['cells'],'rows':result['audit']['rows'],'support_H':result['core']['support.H']['mean'],'support_I':result['core']['support.I']['mean'],'support_delta':result['core']['support.delta']['mean'],'support_p_positive':result['core']['support.p_positive']['mean'],'support_p_negative':result['core']['support.p_negative']['mean'],'support_p_zero':result['core']['support.p_zero']['mean'],'N_minus_NJ_unchanged_scene':result['frame_groups']['N_minus_NJ.unchanged']['scene_equal'],'N_minus_NJ_increased_scene':result['frame_groups']['N_minus_NJ.increased']['scene_equal']},indent=2))
