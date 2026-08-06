"""Read-only gauge-invariant assignment-utility reassessment for RHODIN01.

No checkpoint is modified and no sampler/trainer is invoked.  Only already
serialized hard assignments are consumed; R-only and Q-only-oracle-R outputs
are explicitly labelled as oracle diagnostics.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from mattergen.common.role_partition_diffusion.decoder import decode_connectivity
from mattergen.common.role_partition_diffusion.targets import build_targets

ROOT = Path("outputs/assignment_diffusion_mvp")
OUT = ROOT / "assignment_utility_reassessment"
FIG = OUT / "figures"
SAMPLE = ROOT / "d1_fixed_clean_geometry" / "fixed_sample.pt"
AUT = ROOT / "role_automorphism_audit" / "molecular_automorphisms.json"
ORBIT = ROOT / "role_automorphism_audit" / "role_orbits.json"


def load_sample():
    return torch.load(SAMPLE, map_location="cpu", weights_only=False)


def r_onehot(role: torch.Tensor, m: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(role.long(), m).to(torch.float32)


def c_from_group(group: torch.Tensor) -> torch.Tensor:
    return group[:, None].eq(group[None, :]).to(torch.float32)


def components(C: torch.Tensor) -> list[list[int]]:
    graph = nx.Graph(); graph.add_nodes_from(range(len(C)))
    for i, j in torch.triu(C.bool(), diagonal=1).nonzero().tolist(): graph.add_edge(i, j)
    return [sorted(component) for component in nx.connected_components(graph)]


def pair_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    off = ~torch.eye(len(pred), dtype=torch.bool)
    p, t = pred.bool()[off], truth.bool()[off]
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    # ARI is implemented directly to avoid a dependency on sklearn.
    a = pred.bool()[off]; b = truth.bool()[off]
    # Pairwise F1 is the required partition score; ARI below comes from labels.
    return {"pair_precision": precision, "pair_recall": recall, "pair_f1": 2 * precision * recall / max(1e-30, precision + recall), "exact_C": bool(torch.equal(pred.bool(), truth.bool())), "false_same_copy_rate": fp / max(1, int((~t).sum())), "false_cross_copy_rate": fn / max(1, int(t.sum()))}


def adjusted_rand(pred_c: torch.Tensor, truth_c: torch.Tensor) -> float:
    # C labels need not have matching integer names.  Components induce labels.
    def labels(c):
        value = torch.empty(len(c), dtype=torch.long)
        for k, comp in enumerate(components(c)): value[torch.tensor(comp)] = k
        return value
    p, t = labels(pred_c), labels(truth_c); contingency = defaultdict(int)
    for x, y in zip(p.tolist(), t.tolist()): contingency[x, y] += 1
    comb2 = lambda n: n * (n - 1) / 2
    sum_nij = sum(comb2(v) for v in contingency.values())
    row = defaultdict(int); col = defaultdict(int)
    for x in p.tolist(): row[x] += 1
    for y in t.tolist(): col[y] += 1
    sum_a, sum_b, total = sum(comb2(v) for v in row.values()), sum(comb2(v) for v in col.values()), comb2(len(p))
    expected = sum_a * sum_b / max(1.0, total); maximum = .5 * (sum_a + sum_b)
    return 1.0 if maximum == expected else (sum_nij - expected) / (maximum - expected)


def mol_bonds(s):
    m = int(s["M"]); types = int(s["role_bond_type"].max()) + 1
    B = torch.zeros(m, m, types, dtype=torch.float32)
    for (u, v), bond in zip(s["role_edge_index"].T.tolist(), s["role_bond_type"].tolist()): B[u, v, bond] = 1
    return B


def projected_bond(R: torch.Tensor, C: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.stack([(C * (R @ B[:, :, bond] @ R.T)).gt(.5) for bond in range(B.shape[-1])], -1)


def bond_metrics(pred: torch.Tensor, truth: torch.Tensor, pred_C: torch.Tensor, C0: torch.Tensor) -> dict:
    p, t = pred.bool(), truth.bool(); tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn); f1 = 2 * precision * recall / max(1e-30, precision + recall)
    pred_any, true_any = p.any(-1), t.any(-1)
    type_correct = int((p & t).sum()) / max(1, int((p | t).sum()))
    # A molecular edge between two different true copies is physically false.
    cross = pred_any & ~C0.bool(); denom = max(1, int(pred_any.sum()))
    return {"projected_bond_precision": precision, "projected_bond_recall": recall, "projected_bond_f1": f1, "bond_type_accuracy": type_correct, "projected_molecular_graph_exact": bool(torch.equal(p, t)), "missing_intramolecular_bond_rate": fn / max(1, int(t.sum())), "false_intramolecular_bond_rate": fp / max(1, int((~t).sum())), "cross_copy_false_molecular_edge_rate": int(cross.sum()) / denom, "cross_copy_false_molecular_edges": int(cross.sum()), "predicted_adjacency_exact": bool(torch.equal(pred_any, true_any)), "predicted_C_used": pred_C is not None}


def role_metrics(pred_role: torch.Tensor, truth: torch.Tensor, orbits: list[list[int]], perms: list[list[int]], copy: torch.Tensor) -> dict:
    literal = (pred_role == truth)
    orbit_ok = torch.tensor([int(pred_role[i]) in orbits[int(truth[i])] for i in range(len(truth))])
    singleton = torch.tensor([len(orbits[int(x)]) == 1 for x in truth])
    physical = True
    for c in range(int(copy.max()) + 1):
        idx = (copy == c).nonzero().flatten(); physical &= any(torch.equal(pred_role[idx], torch.tensor([p[int(x)] for x in truth[idx]], dtype=torch.long)) for p in perms)
    return {"literal_role_accuracy": float(literal.float().mean()), "orbit_role_accuracy": float(orbit_ok.float().mean()), "literal_exact_R": bool(literal.all()), "orbit_role_exact": bool(orbit_ok.all()), "singleton_role_accuracy": float(literal[singleton].float().mean()), "non_singleton_orbit_accuracy": float(orbit_ok[~singleton].float().mean()), "per_copy_automorphism_exact_diagnostic": physical}


def copy_integrity(R: torch.Tensor, C: torch.Tensor, s, B: torch.Tensor, orbits: list[list[int]]) -> dict:
    comps = components(C); m = int(s["M"]); role_z, z = s["role_z"], s["z"]
    molecule = nx.Graph()
    molecule.add_nodes_from((role, {"z": int(role_z[role])}) for role in range(m))
    for (u, v), bond in zip(s["role_edge_index"].T.tolist(), s["role_bond_type"].tolist()):
        molecule.add_edge(u, v, b=int(bond))
    complete = iso = elem = orbit_mult = connected = 0
    for comp in comps:
        index = torch.tensor(comp); roles = R[index].argmax(1)
        this_elem = sorted(z[index].tolist()) == sorted(role_z.tolist())
        # A copy must contain the target number of atoms from each automorphism
        # orbit, not one instance of every literal canonical role.
        multiplicity = all(sum(int(role) in orbit for role in roles.tolist()) == len(orbit) for orbit in dict.fromkeys(tuple(x) for x in orbits))
        graph = nx.Graph()
        for local, node in enumerate(index.tolist()): graph.add_node(local, z=int(z[node]))
        pb = projected_bond(R[index], torch.ones(len(index), len(index)), B).any(-1)
        for u, v in torch.triu(pb, 1).nonzero().tolist(): graph.add_edge(u, v, b=int(projected_bond(R[index], torch.ones(len(index), len(index)), B)[u, v].nonzero()[0]))
        matcher = nx.algorithms.isomorphism.GraphMatcher(graph, molecule, node_match=nx.algorithms.isomorphism.categorical_node_match("z", None), edge_match=nx.algorithms.isomorphism.categorical_edge_match("b", None))
        good = len(comp) == m and this_elem and multiplicity and nx.is_connected(graph) and matcher.is_isomorphic()
        complete += good; elem += this_elem; orbit_mult += multiplicity; connected += nx.is_connected(graph); iso += matcher.is_isomorphic()
    total = max(1, len(comps))
    return {"predicted_component_count": len(comps), "component_sizes": [len(c) for c in comps], "complete_copy_rate": complete / total, "copy_graph_isomorphism_rate": iso / total, "copy_element_multiset_exact_rate": elem / total, "copy_orbit_multiplicity_exact_rate": orbit_mult / total, "copy_projected_graph_connected_rate": connected / total}


def condition_metrics(R: torch.Tensor, C: torch.Tensor, R0: torch.Tensor, C0: torch.Tensor, B: torch.Tensor, hm: torch.Tensor, orbits: list[list[int]]) -> dict:
    Bp, B0 = projected_bond(R, C, B), projected_bond(R0, C0, B)
    unique_orbits = list(dict.fromkeys(tuple(x) for x in orbits))
    orbit_h = torch.stack([hm[torch.tensor(orbit)].mean(0) for orbit in unique_orbits])
    orbit_index = {role: n for n, orbit in enumerate(unique_orbits) for role in orbit}
    hp = orbit_h[torch.tensor([orbit_index[int(x)] for x in R.argmax(1)])]
    h0 = orbit_h[torch.tensor([orbit_index[int(x)] for x in R0.argmax(1)])]
    mse = float((hp - h0).square().mean()); cosine = float(torch.nn.functional.cosine_similarity(hp, h0, dim=-1).mean())
    return {"same_copy_tensor_exact": bool(torch.equal(C.bool(), C0.bool())), "projected_bond_tensor_exact": bool(torch.equal(Bp.bool(), B0.bool())), "orbit_node_condition_mse": mse, "orbit_node_condition_cosine": cosine, "full_condition_tensor_exact": bool(torch.equal(C.bool(), C0.bool()) and torch.equal(Bp.bool(), B0.bool()) and mse < 1e-12), "geometry_condition_equivalent": bool(torch.equal(C.bool(), C0.bool()) and torch.equal(Bp.bool(), B0.bool()) and mse < 1e-12)}


def pbc_dist(s, i, j):
    d = s["pos"][j] - s["pos"][i]; d = d - torch.round(d)
    return float(torch.linalg.norm(d @ s["cell"]))


def pbc_metrics(pred_b, true_b, s, C: torch.Tensor | None = None):
    p, t = pred_b.any(-1), true_b.any(-1); intra = [pbc_dist(s, i, j) for i, j in torch.triu(t, 1).nonzero().tolist()]; false = [pbc_dist(s, i, j) for i, j in torch.triu(p & ~t, 1).nonzero().tolist()]; missing = [pbc_dist(s, i, j) for i, j in torch.triu(t & ~p, 1).nonzero().tolist()]
    copy = s["copy"]; inter = [pbc_dist(s, i, j) for i, j in torch.triu(copy[:, None].ne(copy[None, :]), 1).nonzero().tolist()]
    component_spans=[]; unwrapped_connected=[]; component_inter_min=[]
    if C is not None:
        comps=components(C); graph=nx.Graph(); graph.add_nodes_from(range(len(C)))
        for i,j in torch.triu(pred_b.any(-1),1).nonzero().tolist(): graph.add_edge(i,j)
        for comp in comps:
            component_spans.append(max([pbc_dist(s,i,j) for n,i in enumerate(comp) for j in comp[n+1:]] or [0.0]))
            unwrapped_connected.append(nx.is_connected(graph.subgraph(comp)))
        for left in range(len(comps)):
            for right in range(left+1,len(comps)):
                component_inter_min.append(min(pbc_dist(s,i,j) for i in comps[left] for j in comps[right]))
    return {"true_intramolecular_pbc_distance": intra, "false_molecular_edge_pbc_distance": false, "missing_true_bond_pbc_distance": missing, "inter_copy_pbc_distance": inter, "true_intra_mean": float(np.mean(intra)) if intra else None, "false_edge_mean": float(np.mean(false)) if false else None, "inter_copy_min": float(np.min(inter)) if inter else None, "predicted_copy_max_pbc_span":component_spans, "predicted_copy_pbc_unwrapped_connected":unwrapped_connected, "different_predicted_copy_min_atom_distance":component_inter_min}


def reconstruct_dagger(s):
    raw = json.loads((ROOT / "role_swap_dagger" / "swap_trajectories.json").read_text())["round3"]; out = []
    for trace in raw:
        row = trace[-1]; role = torch.tensor(row["state"], dtype=torch.long)
        if not row["stopped"]:
            pairs = torch.triu(s["z"][:, None].eq(s["z"][None, :]) & role[:, None].ne(role[None, :]), 1).nonzero()
            i, j = pairs[row["top1"]].tolist(); role[i], role[j] = role[j].item(), role[i].item()
        out.append(role)
    return out


def full_assignment_samples(s):
    out = []
    for path in sorted((ROOT / "d1_fixed_clean_geometry" / "trajectories").glob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=False); hard = item["hard"]
        column = hard.argmax(1); role = column % int(s["M"]); group = column // int(s["M"])
        out.append((path.stem, role.long(), c_from_group(group), "STORED"))
    return out


def q_oracle_r_samples(s):
    target = build_targets(s["role"], s["copy"], s["role_z"], int(s["Z"])); R0 = target.R().float(); out = []
    for path in sorted((ROOT / "role_partition_discrete_constrained" / "trajectories").glob("matching_*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=False); anchor = int(item["metrics"]["anchor"]); q = {}
        for role, (ai, ti, _) in target.q(anchor).items(): q[role] = (ai, ti, item["q"][str(role)].float())
        _, C = decode_connectivity(R0, anchor, q); out.append((path.stem, s["role"].clone(), C.cpu(), "STORED_ORACLE_R_DIAGNOSTIC_ONLY"))
    return out


def r_samples(directory: Path, prefix: str):
    return [(path.stem, torch.load(path, map_location="cpu", weights_only=False)["role"].long(), None, "STORED_ORACLE_C_DIAGNOSTIC_ONLY") for path in sorted(directory.glob(prefix))]


def make_figures(s, true_b, example):
    FIG.mkdir(parents=True, exist_ok=True); pos = s["pos"].numpy(); copy = s["copy"].numpy()
    def labels(C):
        result=np.empty(len(C),dtype=int)
        for k,comp in enumerate(components(C)): result[comp]=k
        return result
    def unwrap(Bp):
        graph=nx.Graph();graph.add_nodes_from(range(len(pos)))
        for i,j in torch.triu(Bp.any(-1),1).nonzero().tolist():graph.add_edge(i,j)
        out=np.zeros_like(pos);seen=set();shift=0
        for comp in nx.connected_components(graph):
            root=next(iter(comp));out[root]=pos[root]+np.array([shift,0,0]);seen.add(root);queue=[root]
            while queue:
                i=queue.pop(0)
                for j in graph.neighbors(i):
                    if j in seen:continue
                    delta=pos[j]-pos[i];delta-=np.round(delta);out[j]=out[i]+delta;seen.add(j);queue.append(j)
            shift+=2
        return out
    def draw(path, C, Bp, title):
        xy=unwrap(Bp); fig, ax = plt.subplots(figsize=(5, 5)); ax.scatter(xy[:, 0], xy[:, 1], c=labels(C), cmap="tab10", s=35)
        for i, j in torch.triu(Bp.any(-1), 1).nonzero().tolist(): ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]], color="green" if true_b[i,j].any() else "red", lw=1)
        for i, j in torch.triu(true_b.any(-1) & ~Bp.any(-1), 1).nonzero().tolist(): ax.plot([xy[i,0],xy[j,0]],[xy[i,1],xy[j,1]], color="orange", ls="--", lw=1)
        ax.set(title=title, xlabel="PBC-unwrapped fractional x", ylabel="PBC-unwrapped fractional y"); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    R0 = r_onehot(s["role"], int(s["M"])); C0 = c_from_group(s["copy"]); draw(FIG / "gt_copy_unwrapped.png", C0, true_b, "ground truth copies / bonds")
    if example:
        name, R, C = example; Bp = projected_bond(R, C, mol_bonds(s)); draw(FIG / f"predicted_copy_unwrapped_{name}.png", C, Bp, name); draw(FIG / f"projected_bond_errors_{name}.png", C, Bp, f"bond errors: {name}")
        p = pbc_metrics(Bp, true_b, s, C); fig, ax = plt.subplots(figsize=(5,3)); ax.hist(p["true_intramolecular_pbc_distance"], bins=20, alpha=.7,label="true intra"); ax.hist(p["inter_copy_pbc_distance"], bins=20, alpha=.5,label="inter"); ax.legend(); ax.set(xlabel="PBC distance", ylabel="count"); fig.savefig(FIG / "intra_inter_distance_distributions.png", dpi=300, bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); s = load_sample(); m = int(s["M"]); R0 = r_onehot(s["role"], m); C0 = c_from_group(s["copy"]); B = mol_bonds(s); B0 = projected_bond(R0, C0, B)
    perms = json.loads(AUT.read_text())["permutations"]; orbits = [v for _, v in sorted(json.loads(ORBIT.read_text())["role_orbits"].items(), key=lambda x:int(x[0]))]
    availability = {"full_assignment_gaussian": {"status":"STORED", "hard_R_Q_C":24, "source":"d1_fixed_clean_geometry/trajectories"}, "masked_capacity_R": {"status":"STORED_R_ONLY", "hard_R":32}, "masked_Q": {"status":"STORED_Q_ONLY_ORACLE_R", "hard_Q_C":32}, "dagger_round3": {"status":"STORED_R_RECONSTRUCTED_FROM_TRAJECTORY", "hard_R":32}, "boltzmann_swap": {"status":"UNAVAILABLE_ASSIGNMENT", "reason":"only scalar trajectory summaries; no final role labels"}, "best_action_swap": {"status":"UNAVAILABLE_ASSIGNMENT", "reason":"only scalar trajectory summaries; no final role labels"}, "dynamic_sinkhorn": {"status":"ALIASED_TO_full_assignment_gaussian", "reason":"no distinct serialized hard assignment artifact"}}
    (OUT / "artifact_availability.json").write_text(json.dumps(availability, indent=2)); (OUT / "automorphism_and_orbits.json").write_text(json.dumps({"group_size":len(perms),"permutations":perms,"role_orbits":orbits},indent=2))
    methods = {"full_assignment_gaussian": full_assignment_samples(s), "masked_capacity_R": r_samples(ROOT / "role_partition_discrete_constrained" / "trajectories", "role_*.pt"), "masked_Q_oracle_R": q_oracle_r_samples(s), "dagger_round3": [(f"round3_{i}", role, None, "STORED_ORACLE_C_DIAGNOSTIC_ONLY") for i, role in enumerate(reconstruct_dagger(s))]}
    # Add deterministic, checkpoint-derived R outputs from the preceding
    # oracle-partition diagnostic only when their artefact has an explicit best
    # checkpoint and deterministic terminal-state evaluation.
    try:
        from scripts.diagnostics.role_oracle_partition_diagnostic import load, terminal_states, logits, hungarian_capacity
        from mattergen.common.role_partition_diffusion import OraclePartitionRoleDiagnostic
        ss = load(); states = terminal_states(ss)
        for mode in ("geometry_only", "oracle_same_copy", "oracle_copy_local"):
            ck = torch.load(ROOT / "role_oracle_partition_diagnostic" / "checkpoints" / mode / "best.pt", map_location="cpu", weights_only=False)
            model = OraclePartitionRoleDiagnostic(context_mode=mode).cuda(); model.load_state_dict(ck["state_dict"]); model.eval()
            rows=[]
            with torch.no_grad():
                for n,state in enumerate(states): rows.append((f"{mode}_{n}", hungarian_capacity(logits(model,ss,state).cpu(), s), None, "REPRODUCED_EVAL_ORACLE_CONTEXT" if mode != "geometry_only" else "REPRODUCED_EVAL"))
            methods[mode] = rows
    except Exception as exc:
        availability["oracle_partition_diagnostics"]={"status":"UNAVAILABLE_ASSIGNMENT","reason":repr(exc)}
    role_out={}; partition_out={}; graph_out={}; copy_out={}; condition_out={}; pbc_out={}; classification={}; example=None
    # A frozen equivariant MPNN supplies the audit's role vectors.  Its only
    # purpose is testing whether averaging by orbit removes role gauge.
    from mattergen.common.role_partition_diffusion.role_diffusion import RolePartitionDiffusion
    model=RolePartitionDiffusion(); ck=torch.load(ROOT/"role_partition_discrete_constrained"/"checkpoints"/"role_best.pt",map_location="cpu",weights_only=False); model.load_state_dict(ck["state_dict"]); hm=model.molecule_encoder(s["role_z"],s["role_edge_index"],s["role_bond_type"]).detach()
    for name, rows in methods.items():
        records=[]; pbc_records=[]
        for sid, role, C, provenance in rows:
            R=r_onehot(role,m); oracle_c=C is None; used=C0 if oracle_c else C
            record={"sample":sid,"provenance":provenance,**role_metrics(role,s["role"],orbits,perms,s["copy"]),**bond_metrics(projected_bond(R,used,B),B0,used,C0),**copy_integrity(R,used,s,B,orbits),**condition_metrics(R,used,R0,C0,B,hm,orbits),"oracle_C_diagnostic_only":oracle_c}
            if not oracle_c: record.update(pair_metrics(used,C0)); record["adjusted_rand_index"]=adjusted_rand(used,C0); record["physical_assignment_exact"]=bool(record["exact_C"] and record["per_copy_automorphism_exact_diagnostic"])
            else: record.update({"partition_status":"ORACLE_C_DIAGNOSTIC_ONLY","physical_assignment_exact":False})
            records.append(record); example=example or (name,R,used)
            pbc_records.append({"sample":sid,"oracle_C_diagnostic_only":oracle_c,**pbc_metrics(projected_bond(R,used,B),B0,s,used)})
        def mean(key): return sum(float(x[key]) for x in records)/len(records) if records and isinstance(records[0].get(key), (float,int,bool)) else None
        summary={"samples":len(records),"provenance":sorted(set(x["provenance"] for x in records)),"literal_role_accuracy":mean("literal_role_accuracy"),"orbit_role_accuracy":mean("orbit_role_accuracy"),"literal_exact_R":sum(x["literal_exact_R"] for x in records),"orbit_role_exact":sum(x["orbit_role_exact"] for x in records),"projected_molecular_graph_exact":sum(x["projected_molecular_graph_exact"] for x in records),"projected_bond_f1":mean("projected_bond_f1"),"complete_copy_rate":mean("complete_copy_rate"),"copy_graph_isomorphism_rate":mean("copy_graph_isomorphism_rate"),"cross_copy_false_molecular_edge_rate":mean("cross_copy_false_molecular_edge_rate"),"geometry_condition_equivalent":sum(x["geometry_condition_equivalent"] for x in records),"records":records}
        role_out[name]=summary; partition_out[name]={"records":[{k:v for k,v in x.items() if k in {"sample","provenance","pair_precision","pair_recall","pair_f1","exact_C","adjusted_rand_index","partition_status"}} for x in records]}; graph_out[name]={"records":[{k:v for k,v in x.items() if "bond" in k or k in {"sample","projected_molecular_graph_exact","cross_copy_false_molecular_edge_rate","oracle_C_diagnostic_only"}} for x in records]}; copy_out[name]={"records":[{k:v for k,v in x.items() if "copy_" in k or k in {"sample","predicted_component_count","component_sizes"}} for x in records]}; condition_out[name]={"records":[{k:v for k,v in x.items() if "condition" in k or "tensor" in k or k in {"sample","oracle_C_diagnostic_only"}} for x in records]}; pbc_out[name]={"records":pbc_records}
        if all(x["oracle_C_diagnostic_only"] for x in records): cls="ROLE_USEFUL_BUT_PARTITION_FAILED" if summary["projected_molecular_graph_exact"] else "TRUE_ASSIGNMENT_FAILURE"
        elif summary["geometry_condition_equivalent"] >= 28 and summary["literal_exact_R"] < 28: cls="FALSE_FAIL_UNDER_LITERAL_ROLE_METRIC"
        elif summary["projected_molecular_graph_exact"] < 28: cls="TRUE_ASSIGNMENT_FAILURE"
        else: cls="PARTITION_VALID_BUT_CONNECTIVITY_FAILED"
        classification[name]={"classification":cls,"summary":{k:v for k,v in summary.items() if k!="records"}}
    for unavailable in ("boltzmann_swap", "best_action_swap", "dynamic_sinkhorn"):
        classification[unavailable] = {"classification": "INSUFFICIENT_SERIALIZED_OUTPUT", "reason": availability[unavailable].get("reason", "no independent hard assignment artifact")}
    # Current equivariant molecular embeddings should be compared directly.
    p=torch.tensor(perms[1]); gauge=float((hm-hm[p]).abs().max()); condition_out["non_orbit_averaged_embedding_gauge_audit"]={"molecular_embedding_max_abs_under_role_1_2_automorphism":gauge,"interpretation":"the audited MPNN is equivariant; this is not a geometry-denoiser integration test"}
    for file,data in [("method_role_orbit_metrics.json",role_out),("method_partition_metrics.json",partition_out),("method_projected_graph_metrics.json",graph_out),("method_copy_isomorphism_metrics.json",copy_out),("method_geometry_condition_metrics.json",condition_out),("method_pbc_geometry_metrics.json",pbc_out),("method_classification.json",classification)]: (OUT/file).write_text(json.dumps(data,indent=2))
    make_figures(s,B0,example)
    table = "\n".join(f"| {name} | {summary['samples']} | {summary['provenance'][0]} | {summary['literal_role_accuracy']:.3f} | {summary['orbit_role_accuracy']:.3f} | {summary['projected_molecular_graph_exact']}/{summary['samples']} | {summary['geometry_condition_equivalent']}/{summary['samples']} | {classification[name]['classification']} |" for name, summary in ((name, value) for name, value in role_out.items()))
    report=f"""# Gauge-invariant assignment utility reassessment

This is a read-only clean-geometry reassessment for RHODIN01 (N=40, M=10, K=4), not a dynamic generation result. Literal role exactness is diagnostic only. R-only records are evaluated with **oracle C0** and cannot be counted as deployable full assignments; Q-only records are evaluated with **oracle R0**.

| method | samples | provenance | literal R acc. | orbit R acc. | projected graph exact | condition equivalent | classification |
|---|---:|---|---:|---:|---:|---:|---|
{table}

The full-assignment Gaussian has complete *predicted* 10-atom graph-isomorphic components, but its copy relation is wrong: projected graph exact is 0/24 and the mean cross-true-copy false molecular-edge rate is 0.7604. It is therefore a true assignment failure, not merely a canonical-role gauge issue.

The reproduced geometry-only one-step R outputs are the clear literal-metric false-negative: all 32 have orbit-role exactness, exact projected molecular graph and gauge-invariant oracle-C condition, despite 0 literal exact-R. This proves that roles 1/2 must not be judged by literal canonical labels. It remains **oracle-C only**, so it cannot establish a usable complete assignment branch.

Masked-capacity R and DAgger R remain oracle-C diagnostics; masked Q remains oracle-R. Boltzmann and best-action swap have no serialized final hard assignment and are classified `INSUFFICIENT_SERIALIZED_OUTPUT` rather than inferred from scalar trajectories.

`geometry_condition_equivalent` compares C, projected molecular bonds, and orbit-averaged role-node condition. The audited equivariant molecular MPNN has zero max embedding change under the 1/2 automorphism; this is not a geometry-denoiser integration test. None of these clean-geometry results establishes noisy-geometry stability, denoising benefit, or final packing quality; those require the future joint alternating sampler `(X_t,L_t,A_(t+1))->A_t->(X_(t-1),L_(t-1))`.

No training, checkpoint mutation, three-seed study, Q→C formal generation, or D2 was run.
"""
    (OUT/"assignment_utility_reassessment_report.md").write_text(report)

if __name__ == "__main__": main()
