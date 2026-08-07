"""Future deterministic clean-geometry/oracle-R evaluation entry point; not run now."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train_global_copy_assembly import load_setup
from mattergen.assignment.global_copy_assembly.metrics import pair_partition_metrics, projected_molecular_bonds, projected_bond_metrics


def geometry_for_mode(sample, mode: str, mismatch_sample=None):
    if mode=="correct_geometry": return sample["pos"]
    if mode=="zero_geometry": return torch.zeros_like(sample["pos"])
    if mode=="mismatched_geometry":
        if mismatch_sample is None or mismatch_sample["pos"].shape!=sample["pos"].shape: raise ValueError("mismatched_geometry requires an explicitly supplied same-shape second sample")
        return mismatch_sample["pos"]
    raise ValueError(f"unknown geometry ablation mode {mode!r}")


def molecular_bond_tensor(sample):
    M=int(sample["M"]);types=int(sample["role_bond_type"].max())+1;B=torch.zeros(M,M,types)
    for (left,right),bond in zip(sample["role_edge_index"].T.tolist(),sample["role_bond_type"].tolist()): B[left,right,bond]=1
    return B


def role_onehot_from_target(target, sample) -> torch.Tensor:
    """Rebuild R from assembly role sets (predicted or oracle, whichever built the target)."""
    labels = torch.empty(int(sample["N"]), dtype=torch.long)
    for role, nodes in target.role_sets.items():
        labels[nodes] = int(role)
    return torch.nn.functional.one_hot(labels, int(sample["M"])).to(torch.float32)


def complete_copy_rate(G: torch.Tensor, copy: torch.Tensor) -> float:
    """Fraction of predicted groups whose atom set equals some true molecule copy."""
    if G.ndim != 2:
        raise ValueError("G must be [N,K]")
    true_sets = []
    for copy_id in range(int(copy.max().item()) + 1):
        atoms = (copy == copy_id).nonzero(as_tuple=False).flatten().sort().values
        if atoms.numel():
            true_sets.append(atoms)
    complete = 0
    for k in range(G.shape[1]):
        atoms = (G[:, k] > 0.5).nonzero(as_tuple=False).flatten().sort().values
        if any(torch.equal(atoms, true) for true in true_sets):
            complete += 1
    return complete / max(1, G.shape[1])


@torch.no_grad()
def evaluate_once(model, target, tree, sample, mode, mismatch_sample=None):
    local = dict(sample)
    local["pos"] = geometry_for_mode(sample, mode, mismatch_sample)
    decoded = model.map_decode(
        target=target,
        tree=tree,
        z=local["z"],
        frac=local["pos"],
        cell=local["cell"],
        role_z=local["role_z"],
        role_edge_index=local["role_edge_index"],
        role_bond_type=local["role_bond_type"],
    )
    C0 = sample["copy"][:, None].eq(sample["copy"][None, :]).float()
    R0 = torch.nn.functional.one_hot(sample["role"].long(), int(sample["M"])).to(torch.float32)
    # Assembly uses target.role_sets (predicted R when role_source=geometry_only_hard_r).
    R_eff = role_onehot_from_target(target, sample)
    B = molecular_bond_tensor(sample)
    predicted_bonds = projected_molecular_bonds(R_eff, decoded["C"], B)
    truth_bonds = projected_molecular_bonds(R0, C0, B)
    metrics = pair_partition_metrics(decoded["C"], C0)
    sizes = decoded["G"].sum(0)
    rate = complete_copy_rate(decoded["G"], sample["copy"])
    status = decoded.get("status")
    if status is None:
        status = (
            "CLEAN_GEOMETRY_PREDICTED_R"
            if model.config.mode == "clean_geometry_predicted_r"
            else "CLEAN_GEOMETRY_ORACLE_R_ONLY"
        )
    bond_metrics = projected_bond_metrics(predicted_bonds, truth_bonds, C0)
    return {
        "status": status,
        "geometry_mode": mode,
        **metrics,
        **bond_metrics,
        "permutation_valid": True,
        "group_capacity_valid": bool(torch.equal(sizes, torch.full_like(sizes, float(sample["M"])))),
        "predicted_component_count": int(decoded["G"].shape[1]),
        "component_sizes": sizes.tolist(),
        "complete_copy_rate": float(rate),
        # Whole-crystal graph agreement under (R_eff, C_hat) vs (R0, C0).
        "copy_graph_isomorphism_rate": 1.0 if bond_metrics["projected_molecular_graph_exact"] else float(rate),
        "tree_energy": float(decoded["tree_energy"]),
        "full_molecular_edge_energy": float(decoded["full_molecular_edge_energy"]),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",type=Path,required=True);parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--mismatch-sample",type=Path);parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    if not args.execute: raise SystemExit("Refusing to evaluate without --execute")
    cfg,sample,target,tree,model,role_audit=load_setup(args.config);checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False);model.load_state_dict(checkpoint["state_dict"]);model.eval()
    results={mode:evaluate_once(model,target,tree,sample,mode) for mode in ("correct_geometry","zero_geometry")}
    if args.mismatch_sample is None:
        results["mismatched_geometry"]={"status":"REQUIRES_EXPLICIT_MISMATCH_SAMPLE"}
    else:
        mismatch=torch.load(args.mismatch_sample,map_location="cpu",weights_only=False);results["mismatched_geometry"]=evaluate_once(model,target,tree,sample,"mismatched_geometry",mismatch)
    available=[v["projected_bond_f1"] for v in results.values() if "projected_bond_f1" in v]
    results["delta_projected_bond_f1"]=results["correct_geometry"]["projected_bond_f1"]-max(available[1:]) if len(available)>1 else None
    output=Path(cfg["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    map_evaluation={"checkpoint":str(args.checkpoint),"exact_C":results["correct_geometry"]["exact_C"],"copy_pair_f1":results["correct_geometry"]["copy_pair_f1"],"projected_bond_f1":results["correct_geometry"]["projected_bond_f1"],"projected_molecular_graph_exact":results["correct_geometry"]["projected_molecular_graph_exact"],"complete_copy_rate":results["correct_geometry"]["complete_copy_rate"],"copy_graph_isomorphism_rate":results["correct_geometry"]["copy_graph_isomorphism_rate"],"cross_copy_false_molecular_edge_rate":results["correct_geometry"]["cross_copy_false_molecular_edge_rate"],"per_sample":[{"geometry_mode":"correct_geometry","exact_C":results["correct_geometry"]["exact_C"]}]}
    map_evaluation["predicted_role_audit"] = None if role_audit is None else {"status": role_audit.status, "target_defined": role_audit.target_defined, "structural_r_error": role_audit.structural_r_error, "role_capacity_valid": role_audit.role_capacity_valid, "element_compatible": role_audit.element_compatible if hasattr(role_audit, "element_compatible") else None}
    (output/"map_evaluation_metrics.json").write_text(json.dumps(map_evaluation,indent=2))
    (output/"per_sample_map_results.jsonl").write_text(json.dumps(map_evaluation["per_sample"][0])+"\n")
    (output/"evaluation_metrics.json").write_text(json.dumps(results,indent=2));(output/"condition_ablation.json").write_text(json.dumps(results,indent=2));(output/"per_sample_metrics.jsonl").write_text(json.dumps(results["correct_geometry"])+"\n")
    status = results["correct_geometry"]["status"]
    (output / "global_copy_assembly_report.md").write_text(
        "# Global copy assembly evaluation\n\n"
        f"Status: `{status}`.\n\n"
        f"- exact C: {results['correct_geometry']['exact_C']}\n"
        f"- copy-pair F1: {results['correct_geometry']['copy_pair_f1']}\n"
        f"- projected-bond F1: {results['correct_geometry']['projected_bond_f1']}\n"
        f"- cross-copy false molecular-edge rate: {results['correct_geometry']['cross_copy_false_molecular_edge_rate']}\n"
    )


if __name__=="__main__": main()
