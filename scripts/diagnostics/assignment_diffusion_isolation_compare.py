"""Compare independently-produced baseline/current isolation records."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def compare(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict) and {'shape','dtype','values'}.issubset(a) and {'shape','dtype','values'}.issubset(b):
        av,bv=np.asarray(a['values']),np.asarray(b['values']); delta=np.abs(av-bv); denom=np.maximum(np.maximum(np.abs(av),np.abs(bv)),1e-30)
        return {'path':path,'shape':a['shape'],'dtype':a['dtype'],'baseline_shape':b['shape'],'baseline_dtype':b['dtype'],'bitwise_equal':a['sha256']==b['sha256'],'max_absolute_error':float(delta.max()) if delta.size else 0.0,'max_relative_error':float((delta/denom).max()) if delta.size else 0.0}
    if isinstance(a,dict) and isinstance(b,dict): return {k:compare(a.get(k),b.get(k),f'{path}.{k}'.strip('.')) for k in sorted(set(a)|set(b))}
    if isinstance(a,list) and isinstance(b,list): return [compare(x,y,f'{path}[{i}]') for i,(x,y) in enumerate(zip(a,b))]
    if isinstance(a,dict) or isinstance(b,dict) or isinstance(a,list) or isinstance(b,list): return {'path':path,'equal':a==b}
    return {'path':path,'baseline':a,'current':b,'bitwise_equal':a==b}
def main():
 p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('current');p.add_argument('output');x=p.parse_args();a=json.loads(Path(x.baseline).read_text());b=json.loads(Path(x.current).read_text());keys=['eval_forward','train_forward','gradients','optimizer_parameter_groups','state_dict_keys','rng_before','rng_after'];out={'baseline':a,'current':b,'comparison':{k:compare(a[k],b[k],k) for k in keys},'pass':all(a[k]==b[k] for k in keys) and not b['assignment_module_instantiated'] and not b['assignment_trajectory_executed'] and not b['assignment_loss_key_present'] and not b['assignment_parameter_in_optimizer'] and not b['assignment_parameter_gradient']};Path(x.output).write_text(json.dumps(out,indent=2));print(out['pass'])
if __name__=='__main__':main()
