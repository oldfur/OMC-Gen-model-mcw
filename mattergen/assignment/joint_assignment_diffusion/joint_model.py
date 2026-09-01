"""Unified A-aware GemNet joint model: geometry scores + R/G jump logits."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.denoiser import GemNetTDenoiser, get_chemgraph_from_denoiser_output
from mattergen.property_embeddings import get_property_embeddings

from .conditioning import (
    AssignmentGraphMP,
    ClockEmbedding,
    CopyContextPool,
    OrbitRelationTable,
    OrbitSiteEncoder,
    OrbitSlotCopyContext,
    SpatialEdgeAssignmentFeaturizer,
)
from .jump_heads import (
    GJumpHead,
    RJumpHead,
    compute_move_logits,
    jump_pool_diagnostics,
    logits_to_pi,
    logits_to_rates,
)
from .legal_moves import enumerate_legal_moves
from .schedule import AsyncJumpSchedule
from .state import JointAssignmentState


def scf_hard_weight(
    t_scalar: float,
    *,
    enabled: bool,
    gate: str | None,
    threshold: float = 0.5,
) -> float:
    """Hard assignment→geometry gate. t=1 high noise, t=0 clean."""
    if not enabled:
        return 0.0
    if gate not in ("hard", "hard_050", "hard_threshold"):
        return 1.0
    return 0.0 if float(t_scalar) < float(threshold) else 1.0


@dataclass
class JointModelOutput:
    chemgraph_scores: ChemGraph
    node_hidden: torch.Tensor
    move_logits: dict
    move_rates: dict
    move_pi: dict
    diagnostics: dict


class JointAXLModel(nn.Module):
    """A-conditioned GemNet + jump heads (epoch294 warmstart, new modules trainable)."""

    def __init__(
        self,
        denoiser: GemNetTDenoiser,
        *,
        num_orbits: int,
        hidden: int | None = None,
        schedule: AsyncJumpSchedule | None = None,
        g_copy_context_mode: str = "mean",
        g_relation_detach_trunk: bool = True,
        geometry_assignment_conditioning: bool = True,
        scf_time_gate: str | None = None,
        scf_gate_threshold: float = 0.5,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.hidden = int(hidden or getattr(denoiser, "hidden_dim", 512))
        self.schedule = schedule or AsyncJumpSchedule()
        self.g_copy_context_mode = str(g_copy_context_mode or "mean")
        self.g_relation_detach_trunk = bool(g_relation_detach_trunk)
        self.geometry_assignment_conditioning = bool(geometry_assignment_conditioning)
        self.scf_time_gate = None if scf_time_gate in (None, "", "none", "off") else str(scf_time_gate)
        self.scf_gate_threshold = float(scf_gate_threshold)
        self.orbit_encoder = OrbitSiteEncoder(hidden=self.hidden)
        self.orbit_to_node = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        nn.init.zeros_(self.orbit_to_node[-1].weight)
        nn.init.zeros_(self.orbit_to_node[-1].bias)
        self.clock_emb = ClockEmbedding(dim=64)
        self.clock_to_node = nn.Linear(64, self.hidden)
        nn.init.zeros_(self.clock_to_node.weight)
        nn.init.zeros_(self.clock_to_node.bias)
        self.rho = OrbitRelationTable(num_orbits=num_orbits, dim=32)
        emb_edge = int(getattr(denoiser.gemnet, "emb_size_edge", self.hidden))
        ae = getattr(denoiser.gemnet, "angle_edge_emb", None)
        if ae is not None:
            last = list(ae.modules())[-1]
            if isinstance(last, nn.Linear):
                emb_edge = int(last.out_features)
        self.spatial_edge = SpatialEdgeAssignmentFeaturizer(hidden=emb_edge, rho_dim=32, clock_dim=64)
        self.assign_mp = AssignmentGraphMP(hidden=self.hidden, edge_dim=32)
        self.copy_pool = CopyContextPool(hidden=self.hidden)
        self.orbit_slot_ctx = OrbitSlotCopyContext(hidden=self.hidden)
        self.r_head = RJumpHead(hidden=self.hidden, rho_dim=32)
        self.g_head = GJumpHead(hidden=self.hidden, copy_context_mode=self.g_copy_context_mode)
        self.copy_to_node = nn.Linear(self.hidden, self.hidden)
        nn.init.zeros_(self.copy_to_node.weight)
        nn.init.zeros_(self.copy_to_node.bias)

    def new_module_parameters(self):
        for m in (
            self.orbit_encoder,
            self.orbit_to_node,
            self.clock_emb,
            self.clock_to_node,
            self.rho,
            self.spatial_edge,
            self.assign_mp,
            self.copy_pool,
            self.orbit_slot_ctx,
            self.r_head,
            self.g_head,
            self.copy_to_node,
        ):
            yield from m.parameters()

    def pretrained_parameters(self):
        yield from self.denoiser.parameters()

    def g_specific_parameters(self):
        """G-only modules that L_G is allowed to update under isolation."""
        for m in (self.orbit_slot_ctx, self.g_head):
            yield from m.parameters()

    def shared_trunk_parameters(self):
        """GemNet + shared A-conditioning (must not receive L_G when detached)."""
        for m in (
            self.denoiser,
            self.orbit_encoder,
            self.orbit_to_node,
            self.clock_emb,
            self.clock_to_node,
            self.rho,
            self.spatial_edge,
            self.assign_mp,
            self.copy_pool,
            self.copy_to_node,
        ):
            yield from m.parameters()

    def scf_assignment_weight(self, t_scalar: float) -> float:
        """Hard SCF gate: 0 below threshold, 1 at/above. Off if SCF itself is disabled."""
        return scf_hard_weight(
            t_scalar,
            enabled=bool(self.geometry_assignment_conditioning),
            gate=self.scf_time_gate,
            threshold=self.scf_gate_threshold,
        )

    def set_orbit_relations(self, partition, role_edge_index, role_bond_type):
        self.rho.set_from_role_graph(
            partition=partition, role_edge_index=role_edge_index, role_bond_type=role_bond_type
        )

    def _build_a_feedback(
        self,
        state: JointAssignmentState,
        *,
        t_scalar: float,
    ) -> tuple[dict, dict]:
        """Build soft_c_feedback-compatible conditioning dict for GemNet.

        ``state`` is caller-supplied. Trajectory-oracle passes
        ``A_t = forward_CTMC(GT A_0).state_at(t)``. Clean-G passes
        ``A_0^GT`` at every t. This path never reads GJumpHead / RJumpHead outputs.

        ``scf_time_gate='hard'`` applies a config threshold (default 0.5):
        w(t)=0 for t < threshold (GemNet identical to original), w=1 otherwise.
        """
        z_orbit = self.orbit_encoder(state.element_by_orbit.to(state.A.device))
        orbit_of = state.orbit_of()
        copy_of = state.copy_of()
        C = state.C()
        beta_r = float(self.schedule.beta_r(t_scalar).item())
        beta_g = float(self.schedule.beta_g(t_scalar).item())
        lock = float(int(self.schedule.is_r_locked(t_scalar)) + 2 * int(self.schedule.is_g_locked(t_scalar)))
        t_t = torch.tensor(t_scalar, device=state.A.device)
        clock = self.clock_emb(
            t_t.reshape(1),
            torch.tensor(beta_r, device=state.A.device).reshape(1),
            torch.tensor(beta_g, device=state.A.device).reshape(1),
            torch.tensor(lock, device=state.A.device).reshape(1),
        ).squeeze(0)

        node_delta = self.orbit_to_node(z_orbit[orbit_of]) + self.clock_to_node(clock).unsqueeze(0).expand(
            state.N, -1
        )

        def edge_adapter(m, edge_index, cell_offsets):
            delta = self.spatial_edge(
                edge_index=edge_index,
                C=C,
                orbit_of=orbit_of,
                rho=self.rho,
                clock=clock,
            )
            return delta

        def mid_block_node_fn(h, block_idx):
            # nonlocal same-copy MP residual
            delta = self.assign_mp(
                h, copy_of=copy_of, orbit_of=orbit_of, z_orbit=z_orbit, rho=self.rho
            )
            # copy context residual
            _v, c_i, _va = self.copy_pool(h, orbit_of, copy_of, z_orbit, state.K)
            return delta + self.copy_to_node(c_i)

        w = self.scf_assignment_weight(t_scalar)
        if w == 0.0:
            # Strict original GemNet path: do not even register adapters.
            def edge_adapter(m, edge_index, cell_offsets):
                return torch.zeros_like(m)

            def mid_block_node_fn(h, block_idx):
                return torch.zeros_like(h)

            node_delta = torch.zeros_like(node_delta)
        else:
            node_delta = node_delta * w

            _edge = edge_adapter
            _mid = mid_block_node_fn

            def edge_adapter(m, edge_index, cell_offsets):
                return _edge(m, edge_index, cell_offsets) * w

            def mid_block_node_fn(h, block_idx):
                return _mid(h, block_idx) * w

        scf_on = bool(self.geometry_assignment_conditioning) and w != 0.0
        scf = {
            "enabled": scf_on,
            "node_delta": node_delta,
            "edge_adapter": edge_adapter,
            "mid_block_node_fn": mid_block_node_fn,
        }
        if not scf_on:
            scf = {"enabled": False}
        meta = {
            "beta_r": beta_r,
            "beta_g": beta_g,
            "t": t_scalar,
            "z_orbit": z_orbit,
            "orbit_of": orbit_of,
            "copy_of": copy_of,
            "C": C,
            "clock": clock,
            "geometry_assignment_conditioning": bool(self.geometry_assignment_conditioning),
            "scf_time_gate": self.scf_time_gate,
            "scf_gate_threshold": self.scf_gate_threshold,
            "scf_weight": float(w),
            "scf_enabled_runtime": bool(scf_on),
        }
        return scf, meta

    def _denoiser_forward_once(
        self,
        chemgraph: ChemGraph,
        t: torch.Tensor,
        scf: dict,
    ) -> tuple[ChemGraph, torch.Tensor]:
        """Single GemNet path → ChemGraph scores + node_embeddings.

        Mirrors ``GemNetTDenoiser.forward`` but returns node hidden as well so
        jump heads do not require a second full GemNet pass.
        """
        x = chemgraph
        frac_coords, lattice, atom_types, num_atoms, batch = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        t = torch.as_tensor(t, device=lattice.device, dtype=torch.float32).reshape(-1)
        t_enc = self.denoiser.noise_level_encoding(t).to(lattice.device)
        z_per = t_enc
        prop = get_property_embeddings(batch=x, property_embeddings=self.denoiser.property_embeddings)
        if len(prop) > 0:
            z_per = torch.cat([z_per, prop], dim=-1)
        node_condition = None
        if self.denoiser.molecule_conditioner is not None:
            node_condition = self.denoiser.molecule_conditioner(x)
            if node_condition is not None and self.denoiser.molecule_conditioner_gate_center is not None:
                gate = torch.sigmoid(
                    (float(self.denoiser.molecule_conditioner_gate_center) - t.reshape(-1))
                    / float(self.denoiser.molecule_conditioner_gate_width)
                )
                min_scale = float(self.denoiser.molecule_conditioner_gate_min_scale)
                gate = min_scale + (1.0 - min_scale) * gate
                node_condition = node_condition * gate[batch].unsqueeze(-1)
        output = self.denoiser.gemnet(
            z=z_per,
            frac_coords=frac_coords,
            atom_types=atom_types,
            num_atoms=num_atoms,
            batch=batch,
            lengths=None,
            angles=None,
            lattice=lattice,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
            node_condition=node_condition,
            soft_c_feedback=scf,
        )
        h = output.node_embeddings
        pred_atom_types = self.denoiser.fc_atom(h)
        scores = get_chemgraph_from_denoiser_output(
            pred_atom_types=pred_atom_types,
            pred_lattice_eps=output.stress,
            pred_cart_pos_eps=output.forces,
            training=self.denoiser.training,
            element_mask_func=self.denoiser.element_mask_func,
            x_input=x,
        )
        return scores, h

    def forward(
        self,
        chemgraph: ChemGraph,
        t: torch.Tensor,
        state: JointAssignmentState,
        *,
        compute_jumps: bool = True,
    ) -> JointModelOutput:
        # NoiseLevelEncoding expects t.shape == [batch_size], not a 0-dim scalar.
        t = torch.as_tensor(t, device=chemgraph["pos"].device, dtype=torch.float32).reshape(-1)
        t_scalar = float(t[0].item())
        scf, meta = self._build_a_feedback(state, t_scalar=t_scalar)
        # One GemNet only (scores + node_embeddings).
        scores, h = self._denoiser_forward_once(chemgraph, t, scf)
        z_orbit = meta["z_orbit"]
        orbit_of = meta["orbit_of"]
        copy_of = meta["copy_of"]
        move_logits: dict = {}
        move_rates: dict = {}
        move_pi: dict = {}
        v_a = None
        jump_diag: dict = {}
        if compute_jumps:
            v, c_i, v_a = self.copy_pool(h, orbit_of, copy_of, z_orbit, state.K)
            moves = enumerate_legal_moves(state)
            use_slots = self.g_copy_context_mode in (
                "orbit_slot",
                "orbit_slot_geometry",
                "template_counterfactual",
            )
            use_geom = self.g_copy_context_mode in (
                "orbit_slot_geometry",
                "template_counterfactual",
            )
            if self.g_relation_detach_trunk:
                h_g = h.detach()
                z_g = z_orbit.detach()
            else:
                h_g, z_g = h, z_orbit
            scored, slot_diag = compute_move_logits(
                moves=moves,
                h=h,
                state=state,
                z_orbit=z_orbit,
                c_i=c_i,
                v_copies=v,
                rho_table=self.rho,
                r_head=self.r_head,
                g_head=self.g_head,
                slot_ctx=self.orbit_slot_ctx if use_slots else None,
                t_scalar=t_scalar,
                frac=chemgraph["pos"] if use_geom else None,
                cell=chemgraph["cell"] if use_geom else None,
                h_g=h_g,
                z_orbit_g=z_g,
            )
            slot_diag["g_relation_detach_trunk"] = bool(self.g_relation_detach_trunk)
            jump_diag.update(slot_diag)
            move_logits = scored
            # J1.1: r_m = β(t) · softmax(ℓ)_m  (fixed total exit rate)
            move_pi = logits_to_pi(scored)
            move_rates = logits_to_rates(
                scored,
                beta_r=meta["beta_r"],
                beta_g=meta["beta_g"],
            )
            jump_diag.update(
                jump_pool_diagnostics(scored, beta_r=meta["beta_r"], beta_g=meta["beta_g"])
            )
        return JointModelOutput(
            chemgraph_scores=scores,
            node_hidden=h,
            move_logits=move_logits,
            move_rates=move_rates,
            move_pi=move_pi,
            diagnostics={
                **meta,
                "v_A": v_a,
                "num_r_moves": len(move_logits.get("R", [])),
                "num_g_moves": len(move_logits.get("G", [])),
                **jump_diag,
            },
        )
