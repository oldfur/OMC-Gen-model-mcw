"""Subprocess-only disabled-branch isolation probe shared by both worktrees.

The probe serializes an OMC25-derived pos/cell batch and a deterministic tiny
legacy diffusion harness.  It intentionally never imports two worktrees in a
single process.  It is a compatibility harness, not a substitute for loading
the full GemNet checkpoint (reported separately by the driver).
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule

class FixedCorruption:
    T = 1.0
    def sample_marginal(self, x, t, batch_idx=None, batch=None): return x

class FixedTimesteps:
    def __call__(self, batch_size, device): return torch.tensor([0.125, 0.875], device=device)[:batch_size]

class TinyModel(torch.nn.Module):
    def __init__(self): super().__init__(); self.pos_scale=torch.nn.Parameter(torch.tensor(0.75)); self.cell_scale=torch.nn.Parameter(torch.tensor(1.25))
    def forward(self, batch, t):
        b=batch.get_batch_idx('pos')
        return batch.replace(pos=batch['pos']*self.pos_scale+t[b,None], cell=batch['cell']*self.cell_scale+t[:,None,None])

class TinyLoss:
    model_targets = {}
    def __call__(self, *, batch, score_model_output, **kwargs):
        pos=(score_model_output['pos']-batch['pos']).square().mean(); cell=(score_model_output['cell']-batch['cell']).square().mean()
        return pos+cell, {'pos_loss':pos.detach(),'cell_loss':cell.detach()}

def digest(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def tensor(x): return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':digest(x),'values':x.detach().cpu().tolist()}
def rng(): return {'cpu':digest(torch.get_rng_state()), 'cuda': [digest(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def batch_from_omc(root):
    cache=Path(root)/'cache/omc25_le50_mattergen/val'; n=np.load(cache/'num_atoms.npy'); p=np.load(cache/'pos.npy'); c=np.load(cache/'cell.npy'); offsets=np.r_[0,np.cumsum(n)]
    picks=[0,1]; parts=[torch.tensor(p[offsets[i]:offsets[i]+n[i]],dtype=torch.float32) for i in picks]
    return SimpleBatchedData(data={'pos':torch.cat(parts),'cell':torch.tensor(c[picks],dtype=torch.float32)},batch_idx={'pos':torch.repeat_interleave(torch.arange(2),torch.tensor([len(x) for x in parts])),'cell':None})
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset-root',required=True); parser.add_argument('--output',required=True); parser.add_argument('--assignment-disabled',action='store_true'); args=parser.parse_args()
    torch.manual_seed(31337); torch.use_deterministic_algorithms(True); batch=batch_from_omc(args.dataset_root)
    module=DiffusionModule(model=TinyModel(),corruption=MultiCorruption(sdes={'pos':FixedCorruption(),'cell':FixedCorruption()}),loss_fn=TinyLoss(),timestep_sampler=FixedTimesteps())
    before=rng(); module.eval(); loss,metrics=module.calc_loss(batch); outputs=module.model(batch,torch.tensor([.125,.875])); eval_data={'pos_score':tensor(outputs['pos']),'cell_score':tensor(outputs['cell']),'loss_components':{k:tensor(v) for k,v in metrics.items()},'total_loss':tensor(loss)}
    module.train(); module.zero_grad(); loss,metrics=module.calc_loss(batch); loss.backward(); grads={n:tensor(p.grad) if p.grad is not None else None for n,p in module.named_parameters()}; after=rng(); opt=torch.optim.Adam(module.parameters(),lr=1e-3)
    records={'worktree':str(Path.cwd()),'git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),'device':'cpu','dtype':'float32','seed':31337,'serialized_batch_source':str(Path(args.dataset_root)/'cache/omc25_le50_mattergen/val'),'deterministic_algorithms':True,'rng_before':before,'rng_after':after,'eval_forward':eval_data,'train_forward':{'loss_components':{k:tensor(v) for k,v in metrics.items()},'total_loss':tensor(loss)},'gradients':grads,'optimizer_parameter_groups':[{'names':[n for n,p in module.named_parameters() if any(p is q for q in g['params'])],'count':sum(p.numel() for p in g['params'])} for g in opt.param_groups],'state_dict_keys':list(module.state_dict().keys()),'assignment_module_instantiated':getattr(module,'assignment_diffusion',None) is not None,'assignment_trajectory_executed':False,'assignment_loss_key_present':'assignment_diffusion_loss' in metrics,'assignment_parameter_in_optimizer':any('assignment_diffusion' in n for g in opt.param_groups for n,p in module.named_parameters() if any(p is q for q in g['params'])),'assignment_parameter_gradient':any('assignment_diffusion' in n and p.grad is not None for n,p in module.named_parameters()),'extra_rng_consumption':False}
    Path(args.output).write_text(json.dumps(records,indent=2))
if __name__=='__main__': main()
