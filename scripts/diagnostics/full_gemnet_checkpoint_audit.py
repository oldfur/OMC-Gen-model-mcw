"""Full GemNet checkpoint isolation runner; one worktree per process only."""
from __future__ import annotations
import argparse, copy, gzip, hashlib, json, os, random, sys, types
from pathlib import Path
import numpy as np
import torch

# The audit environment has the complete model stack except an annotation-only
# emmet class.  Supply that class locally; it is never part of model execution.
emmet = types.ModuleType("emmet"); emmet_core = types.ModuleType("emmet.core"); material = types.ModuleType("emmet.core.material")
class PropertyOrigin: pass
material.PropertyOrigin = PropertyOrigin
sys.modules.setdefault("emmet", emmet); sys.modules.setdefault("emmet.core", emmet_core); sys.modules.setdefault("emmet.core.material", material)

from hydra.utils import instantiate
from mattergen.diffusion.data.batched_data import SimpleBatchedData

DATA_ROOT = Path("/home/mcw/OMC-Gen-model/datasets/omc25_le50_sinkhorn_subset_3k")

class AuditBatch(SimpleBatchedData):
    def __getattr__(self, key):
        if key in self.data: return self.data[key]
        raise AttributeError(key)

    def replace(self, **vals): return AuditBatch(data=dict(self.data, **vals), batch_idx=self.batch_idx)

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()
def state_hash(x): return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def environment():
    return {"python":sys.executable,"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"device":"cpu","dtype":"float32","deterministic_algorithms":True,"cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,"env":{k:os.environ.get(k) for k in ["CONDA_DEFAULT_ENV","CUDA_VISIBLE_DEVICES","CUBLAS_WORKSPACE_CONFIG","PYTHONPATH"]}}
def rng():
    return {"python":repr(random.getstate()),"numpy":np.random.get_state()[1].tolist(),"torch_cpu":state_hash(torch.get_rng_state()),"torch_cuda":[state_hash(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
def config_from_source(checkpoint):
    raw=torch.load(checkpoint,map_location="cpu"); config=copy.deepcopy(raw["config"]["lightning_module"])
    # Existing trained checkpoints request a historical dynamic assignment
    # implementation absent from both audited source trees.  This keeps the
    # complete GemNet path while producing an explicit initialized checkpoint.
    config["diffusion_module"]["model"]["use_dynamic_sinkhorn_assignment"]=False
    return config, raw
def instantiate_full(checkpoint):
    config, raw=config_from_source(checkpoint)
    model=instantiate(config)
    return model, config, raw
def feature(atom):
    hybrid={"UNSPECIFIED":0,"S":1,"SP":2,"SP2":3,"SP3":4,"SP3D":5,"SP3D2":6}
    return [int(atom.get("atomic_num",0)),min(max(int(atom.get("formal_charge",0))+6,0),12),min(max(int(atom.get("degree",0)),0),11),min(max(int(atom.get("total_valence",0)),0),13),int(bool(atom.get("is_aromatic",False))),hybrid.get(str(atom.get("hybridization","UNSPECIFIED")),7)]
def make_serialized_batches(destination):
    destination.mkdir(parents=True,exist_ok=True); cache=DATA_ROOT/"cache/omc25_le50_mattergen/val"; numbers=np.load(cache/"atomic_numbers.npy"); pos=np.load(cache/"pos.npy"); cell=np.load(cache/"cell.npy"); counts=np.load(cache/"num_atoms.npy"); ids=np.load(cache/"structure_id.npy"); offsets=np.r_[0,np.cumsum(counts)]
    mappings={}
    with gzip.open(DATA_ROOT/"molecule_mapping/omc25_subset_val_molmap_hybrid_v3.jsonl.gz","rt") as f:
        for line in f:
            r=json.loads(line)
            if r.get("success"): mappings[r["material_id"]]=r
    graphs={}
    with gzip.open(DATA_ROOT/"molecule_mapping/oe62_hybrid_graphs_subset.jsonl.gz","rt") as f:
        for line in f:
            r=json.loads(line)
            if r.get("ok") and r.get("transfer_mode")=="rdkit_explicit_h_full_match": graphs[r["refcode_csd"]]=r
    pools={2:[],4:[]}
    for i, identity in enumerate(ids):
        r=mappings.get(str(identity)); m=r and r["mapping"]; z=int(m["num_molecules"]) if m else 0
        if z in pools and r.get("csd_refcode") in graphs and len(m["mol_atom_idx"])==int(counts[i]): pools[z].append((i,r,graphs[r["csd_refcode"]]))
    # Three real batches; each has >1 graph, with Z=2/4 and intentionally varied N/M.
    selections=[pools[2][:2],pools[4][:2], [pools[2][2],pools[4][2]]]
    manifest=[]
    for batch_id, chosen in enumerate(selections):
        data={k:[] for k in ["pos","atomic_numbers","mol_copy_id","mol_atom_id","mol_x","mol_bond_edge_index","mol_bond_attr","mol_bond_d0","cell","num_atoms"]}; batch=[]; names=[]; offset=0
        for graph_id,(i,r,g) in enumerate(chosen):
            n=int(counts[i]); first=int(offsets[i]); m=r["mapping"]; local_pos=torch.tensor(pos[first:first+n],dtype=torch.float32); local_cell=torch.tensor(cell[i],dtype=torch.float32)
            data["pos"].append(local_pos); data["atomic_numbers"].append(torch.tensor(numbers[first:first+n],dtype=torch.long)); data["mol_copy_id"].append(torch.tensor(m["mol_id"],dtype=torch.long)); data["mol_atom_id"].append(torch.tensor(m["mol_atom_idx"],dtype=torch.long)); data["mol_x"].append(torch.tensor([feature(g["atom_features"][int(role)]) for role in m["mol_atom_idx"]],dtype=torch.long))
            edges=[]; attrs=[]; d0=[]
            for bond in r.get("crystal_bonds",[]):
                a,b=int(bond["begin"]),int(bond["end"]); typ=min(max(int(bond.get("type",0)),0),7)
                edges.extend([[offset+a,offset+b],[offset+b,offset+a]]); attrs.extend([[typ,int(typ==4)],[typ,int(typ==4)]])
                delta=local_pos[a]-local_pos[b]; delta=delta-torch.round(delta); length=torch.linalg.norm(delta@local_cell); d0.extend([length,length])
            data["mol_bond_edge_index"].append(torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)); data["mol_bond_attr"].append(torch.tensor(attrs,dtype=torch.long) if attrs else torch.empty((0,2),dtype=torch.long)); data["mol_bond_d0"].append(torch.stack(d0) if d0 else torch.empty(0)); data["cell"].append(local_cell); data["num_atoms"].append(n); batch.append(torch.full((n,),graph_id,dtype=torch.long)); names.append(str(ids[i])); offset+=n
        packed={"data":{k:(torch.stack(v) if k=="cell" else torch.tensor(v,dtype=torch.long) if k=="num_atoms" else torch.cat(v,dim=1 if k=="mol_bond_edge_index" else 0)) for k,v in data.items()},"batch_idx":{"pos":torch.cat(batch),"cell":None},"structure_ids":names}
        torch.save(packed,destination/f"batch_{batch_id}.pt"); manifest.append({"file":f"batch_{batch_id}.pt","structure_ids":names,"N":[int(x) for x in data["num_atoms"]],"Z":[int(x[1]["mapping"]["num_molecules"]) for x in chosen],"M":[len(set(x[1]["mapping"]["mol_atom_idx"])) for x in chosen]})
    (destination/"manifest.json").write_text(json.dumps(manifest,indent=2)); return manifest
def load_batch(path):
    value=torch.load(path,map_location="cpu"); return AuditBatch(data=value["data"],batch_idx=value["batch_idx"]),value["structure_ids"]
def tensors(obj,prefix=""):
    result={}
    if isinstance(obj,torch.Tensor): result[prefix]=obj.detach().cpu()
    elif hasattr(obj,"__dict__"):
        for k,v in vars(obj).items():
            if isinstance(v,torch.Tensor): result[f"{prefix}.{k}".strip(".")]=v.detach().cpu()
    return result
def audit_noise(module,batch,t,cell_noise,pos_noise):
    original=torch.randn_like; queue=[cell_noise,pos_noise]
    def supplied(x,*args,**kwargs):
        if not queue: raise RuntimeError("audit noise queue exhausted")
        noise=queue.pop(0)
        if tuple(x.shape)!=tuple(noise.shape): raise RuntimeError(f"audit noise shape mismatch expected {tuple(x.shape)} got {tuple(noise.shape)}")
        return noise.to(device=x.device,dtype=x.dtype)
    torch.randn_like=supplied
    try: noisy=module.diffusion_module.corruption.sample_marginal(batch,t)
    finally: torch.randn_like=original
    if queue: raise RuntimeError("audit noise queue not fully consumed")
    return noisy
def run(model, batch_path, stochastic_path, mode):
    batch, ids=load_batch(batch_path); stochastic=torch.load(stochastic_path,map_location="cpu"); t=stochastic["t"]
    model.train(mode=="train"); module=model.diffusion_module; hook_data={}
    def capture(name):
        def h(_m,_i,o): hook_data.update({f"{name}.{k}":v.detach().cpu() for k,v in tensors(o).items()})
        return h
    hooks=[module.model.gemnet.register_forward_hook(capture("gemnet"))]
    if module.model.molecule_conditioner is not None: hooks.append(module.model.molecule_conditioner.register_forward_hook(capture("condition")))
    before=rng(); noisy=audit_noise(model,batch,t,stochastic["cell_noise"],stochastic["pos_noise"]); score=module.model(noisy,t); loss,metrics=module.loss_fn(multi_corruption=module.corruption,batch=batch,noisy_batch=noisy,score_model_output=score,t=t,node_is_unmasked=None); after_forward=rng()
    if mode=="train": model.zero_grad(set_to_none=True); loss.backward()
    after=rng()
    for h in hooks:h.remove()
    grads={n:p.grad.detach().cpu() for n,p in module.named_parameters() if p.grad is not None}
    configured=model.configure_optimizers(); optimizer=configured[0][0] if isinstance(configured,tuple) else configured
    names={id(value):name for name,value in module.named_parameters()}
    groups=[]
    for group in optimizer.param_groups:
        groups.append({"names":[names[id(value)] for value in group["params"]],"count":sum(value.numel() for value in group["params"]),"hyperparameters":{key:value for key,value in group.items() if key!="params"}})
    result={"ids":ids,"mode":mode,"t":t.cpu(),"noisy_pos":noisy["pos"].detach().cpu(),"noisy_cell":noisy["cell"].detach().cpu(),"pos_score":score["pos"].detach().cpu(),"cell_score":score["cell"].detach().cpu(),"atomic_numbers_output":score["atomic_numbers"].detach().cpu(),"loss":loss.detach().cpu(),"loss_components":{k:v.detach().cpu() for k,v in metrics.items()},"internals":hook_data,"gradients":grads,"rng_before":before,"rng_after_forward":after_forward,"rng_after":after,"assignment": {"instantiated":getattr(module,"assignment_diffusion",None) is not None,"loss_key":"assignment_diffusion_loss" in metrics,"trajectory":False,"parameters":[n for n,_ in module.named_parameters() if "assignment_diffusion" in n]},"parameter_names":[n for n,_ in module.named_parameters()],"state_keys":list(module.state_dict().keys()),"optimizer_groups":groups}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--source-checkpoint",required=True);p.add_argument("--initialized-checkpoint");p.add_argument("--serialized-dir");p.add_argument("--create-initialized",action="store_true");p.add_argument("--create-stochastic",action="store_true");p.add_argument("--mode",choices=["eval","train"]);p.add_argument("--batch");p.add_argument("--stochastic");p.add_argument("--output");a=p.parse_args(); torch.manual_seed(20260804); random.seed(20260804); np.random.seed(20260804); torch.use_deterministic_algorithms(True)
    if a.create_initialized:
        model,config,raw=instantiate_full(a.source_checkpoint); payload={"state_dict":model.state_dict(),"config":config,"source_checkpoint":a.source_checkpoint,"parameter_count":sum(p.numel() for p in model.parameters())}; torch.save(payload,a.initialized_checkpoint); print(json.dumps({"initialized_checkpoint":a.initialized_checkpoint,"sha256":sha(a.initialized_checkpoint),"parameter_count":payload["parameter_count"],"source_state_keys":len(raw["state_dict"]),"initialized_state_keys":len(payload["state_dict"]),"environment":environment()},indent=2)); return
    if a.serialized_dir:
        directory=Path(a.serialized_dir)
        if a.create_stochastic:
            manifest=json.loads((directory/"manifest.json").read_text())
            for index,item in enumerate(manifest):
                batch,_=load_batch(directory/item["file"]); generator=torch.Generator().manual_seed(7100+index)
                torch.save({"t":torch.tensor([0.17,0.73],dtype=torch.float32),"pos_noise":torch.randn(batch.pos.shape,generator=generator),"cell_noise":torch.randn(batch.cell.shape,generator=generator)},directory/f"stochastic_{index}.pt")
            print(json.dumps({"created_stochastic":len(manifest),"seed_base":7100},indent=2)); return
        print(json.dumps(make_serialized_batches(directory),indent=2)); return
    model,config,_=instantiate_full(a.source_checkpoint); initial=torch.load(a.initialized_checkpoint,map_location="cpu"); loaded=model.load_state_dict(initial["state_dict"],strict=True); output=run(model,Path(a.batch),Path(a.stochastic),a.mode); output["load"]={"missing":list(loaded.missing_keys),"unexpected":list(loaded.unexpected_keys),"checkpoint_sha256":sha(a.initialized_checkpoint),"parameter_count":sum(p.numel() for p in model.parameters()),"environment":environment()}; torch.save(output,a.output)
if __name__=="__main__":main()
