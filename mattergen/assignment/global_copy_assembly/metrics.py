"""Gauge-invariant clean-oracle-R metrics for future evaluation commands."""
from __future__ import annotations

import torch


def pair_partition_metrics(C: torch.Tensor, C0: torch.Tensor) -> dict[str,float|bool]:
    if C.shape != C0.shape or C.ndim != 2 or C.shape[0] != C.shape[1]: raise ValueError("C and C0 must be equally shaped square matrices")
    off=~torch.eye(len(C),dtype=torch.bool,device=C.device);p=C.bool()[off];t=C0.bool()[off]
    tp=(p&t).sum().item();fp=(p&~t).sum().item();fn=(~p&t).sum().item();precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return {"copy_pair_precision":precision,"copy_pair_recall":recall,"copy_pair_f1":2*precision*recall/max(1e-30,precision+recall),"exact_C":bool(torch.equal(C.bool(),C0.bool()))}


def projected_molecular_bonds(R0: torch.Tensor, C: torch.Tensor, molecular_bonds: torch.Tensor) -> torch.Tensor:
    if R0.ndim!=2 or molecular_bonds.ndim!=3: raise ValueError("R0[N,M] and molecular_bonds[M,M,B] are required")
    if C.shape!=(len(R0),len(R0)) or molecular_bonds.shape[:2]!=(R0.shape[1],R0.shape[1]): raise ValueError("projected-bond shapes are incompatible")
    return torch.stack([(C*(R0@molecular_bonds[:,:,bond]@R0.T)).gt(.5) for bond in range(molecular_bonds.shape[-1])],-1)


def projected_bond_metrics(predicted: torch.Tensor, truth: torch.Tensor, C0: torch.Tensor) -> dict[str,float|bool]:
    if predicted.shape!=truth.shape: raise ValueError("predicted and truth molecular bond tensors must have equal shape")
    p,t=predicted.bool(),truth.bool();tp=(p&t).sum().item();fp=(p&~t).sum().item();fn=(~p&t).sum().item();precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    pred_any=p.any(-1);cross=pred_any&~C0.bool()
    return {"projected_bond_precision":precision,"projected_bond_recall":recall,"projected_bond_f1":2*precision*recall/max(1e-30,precision+recall),"projected_molecular_graph_exact":bool(torch.equal(p,t)),"cross_copy_false_molecular_edge_rate":cross.sum().item()/max(1,pred_any.sum().item())}
