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
    ):
        super().__init__()
        self.denoiser = denoiser
        self.hidden = int(hidden or getattr(denoiser, "hidden_dim", 512))
        self.schedule = schedule or AsyncJumpSchedule()
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
        self.r_head = RJumpHead(hidden=self.hidden, rho_dim=32)
        self.g_head = GJumpHead(hidden=self.hidden)
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
            self.r_head,
            self.g_head,
            self.copy_to_node,
        ):
            yield from m.parameters()

    def pretrained_parameters(self):
        yield from self.denoiser.parameters()

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
        """Build soft_c_feedback-compatible conditioning dict for GemNet."""
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

        scf = {
            "enabled": True,
            "node_delta": node_delta,
            "edge_adapter": edge_adapter,
            "mid_block_node_fn": mid_block_node_fn,
        }
        meta = {
            "beta_r": beta_r,
            "beta_g": beta_g,
            "t": t_scalar,
            "z_orbit": z_orbit,
            "orbit_of": orbit_of,
            "copy_of": copy_of,
            "C": C,
            "clock": clock,
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
            scored = compute_move_logits(
                moves=moves,
                h=h,
                state=state,
                z_orbit=z_orbit,
                c_i=c_i,
                v_copies=v,
                rho_table=self.rho,
                r_head=self.r_head,
                g_head=self.g_head,
            )
            move_logits = scored
            # J1.1: r_m = β(t) · softmax(ℓ)_m  (fixed total exit rate)
            move_pi = logits_to_pi(scored)
            move_rates = logits_to_rates(
                scored,
                beta_r=meta["beta_r"],
                beta_g=meta["beta_g"],
            )
            jump_diag = jump_pool_diagnostics(
                scored, beta_r=meta["beta_r"], beta_g=meta["beta_g"]
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
