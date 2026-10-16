#!/usr/bin/env python3
"""Verify packaged results against frozen regression outputs and earlier archived summaries."""
from pathlib import Path
import json, math, platform, sys
import reproduce
HERE=Path(__file__).resolve().parent
result=reproduce.calc()
reproduce.compare(result,json.loads((HERE/'expected_results.json').read_text(encoding='utf-8')))
checks=[]
def check(name,observed,reference):
    ok=math.isclose(observed,reference,rel_tol=1e-10,abs_tol=1e-12)
    checks.append({'name':name,'observed':observed,'reference':reference,'absolute_difference':abs(observed-reference),'pass':ok})
    assert ok,(name,observed,reference)
a=json.loads((HERE/'references/nj_frozen_aggregate_20261005.json').read_text(encoding='utf-8-sig'))
b=json.loads((HERE/'references/supplementary_statistics_20261010.json').read_text(encoding='utf-8-sig'))
for dst,src in [('ade_O','ade_O'),('ade_F','ade_F'),('ade_NJ','ade_NJ'),('ade_N','ade_N'),('R_policy','R_policy'),('R_support','R_support'),('policy.delta','delta_policy'),('support.delta','delta_support')]:
    for stat in ['mean','sample_sd']:
        check('frozen_aggregate.'+dst+'.'+stat,result['core'][dst][stat],a['hierarchy'][src][stat])
for side in ['policy','support']:
    for metric in ['H','I','delta','p_positive','p_negative','p_zero']:
        for dst,src in [('mean','mean'),('sample_sd','seed_sd'),('positive_seeds','positive_seeds'),('negative_seeds','negative_seeds')]:
            check('supplement.'+side+'.'+metric+'.'+dst,result['core'][side+'.'+metric][dst],b['two_sided_mass'][side][metric][src])
for pair,refpair in [('N_minus_NJ','N_vs_NJ_same_run'),('N_minus_O','N_vs_O_same_run')]:
    for group in ['unchanged','increased','decreased']:
        if pair=='N_minus_NJ' and group=='decreased':continue
        observed=result['frame_groups'][pair+'.'+group]; reference=b['frame_strata'][refpair+'/'+group]
        check(pair+'.'+group+'.windows',observed['mean_windows_per_cell'],reference['average_windows'])
        for dst,src in [('window_weighted','window_mean'),('scene_equal','scene_mean')]:
            for stat,refstat in [('mean','mean'),('sample_sd','seed_sd'),('positive_seeds','positive_seeds'),('negative_seeds','negative_seeds')]:
                check(pair+'.'+group+'.'+dst+'.'+stat,observed[dst][stat],reference[src][refstat])
reproduced=HERE/'reproduced_results.json'
reproduced.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
files=['reproduce.py','validate.py','expected_results.json','reproduced_results.json','references/nj_frozen_aggregate_20261005.json','references/supplementary_statistics_20261010.json']
validation={'status':'PASS','validation_date':'2026-10-12','environment':{'python':sys.version,'platform':platform.platform()},'scope':'Author-side reaggregation of archived results, not third-party replication or model inference.','regression_reference':'Entire result compared with the frozen expected_results.json. Independent generation of that complete expected file is not asserted.','reference_check':'Selected results additionally checked against separately retained 2026-10-05 and 2026-10-10 summaries of the same underlying experiments. These references are not independent experiments.','audit':result['audit'],'reference_checks':checks,'files':[{'path':f,'sha256':reproduce.sha(HERE/f)} for f in files]}
(HERE/'validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','cells':result['audit']['cells'],'rows':result['audit']['rows'],'reference_checks':len(checks),'max_reference_abs_difference':max(x['absolute_difference'] for x in checks),'training_or_inference_executed':False},indent=2))
