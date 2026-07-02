from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int, n_layers: int, act: str = "gelu") -> None:
        super().__init__()
        acts = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        self.act = acts[act]
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.linear_pre(x))
        for layer in self.linears:
            x = self.act(layer(x)) + x
        return self.linear_post(x)


class LinearAttention(nn.Module):
    def __init__(self, n_hidden: int, n_head: int, dropout: float) -> None:
        super().__init__()
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        self.n_head = n_head
        self.query = nn.Linear(n_hidden, n_hidden)
        self.key = nn.Linear(n_hidden, n_hidden)
        self.value = nn.Linear(n_hidden, n_hidden)
        self.proj = nn.Linear(n_hidden, n_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        y = x if y is None else y
        bsz, n_query, channels = x.shape
        n_key = y.shape[1]
        q = self.query(x).view(bsz, n_query, self.n_head, channels // self.n_head).transpose(1, 2)
        k = self.key(y).view(bsz, n_key, self.n_head, channels // self.n_head).transpose(1, 2)
        v = self.value(y).view(bsz, n_key, self.n_head, channels // self.n_head).transpose(1, 2)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-1)
        d_inv = 1.0 / ((q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True) + 1.0e-8)
        out = self.drop((q @ (k.transpose(-2, -1) @ v)) * d_inv + q)
        out = out.transpose(1, 2).contiguous().view(bsz, n_query, channels)
        return self.proj(out)


class PhysicsStateAttention(nn.Module):
    def __init__(self, n_hidden: int, n_head: int, slice_num: int, dropout: float) -> None:
        super().__init__()
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_hidden // n_head
        self.scale = self.head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(1, n_head, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(n_hidden, n_hidden)
        self.in_project_fx = nn.Linear(n_hidden, n_hidden)
        self.in_project_slice = nn.Linear(self.head_dim, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.proj = nn.Linear(n_hidden, n_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_token, channels = x.shape
        fx_mid = self.in_project_fx(x).view(bsz, n_token, self.n_head, self.head_dim).transpose(1, 2)
        x_mid = self.in_project_x(x).view(bsz, n_token, self.n_head, self.head_dim).transpose(1, 2)

        slice_logits = self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5.0)
        slice_weights = F.softmax(slice_logits, dim=-1)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm.unsqueeze(-1) + 1.0e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out_slice = self.drop(attn) @ v

        out = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out = out.transpose(1, 2).contiguous().view(bsz, n_token, channels)
        return self.proj(out)


class BoundaryHeatFluxGraphEncoder(nn.Module):
    def __init__(
        self,
        global_branch_size: int,
        boundary_token_count: int,
        n_hidden: int,
        n_head: int,
        mlp_layers: int,
        act: str,
        graph_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = n_hidden // n_head
        self.graph_k = graph_k
        boundary_pos = torch.linspace(0.0, 1.0, boundary_token_count).view(1, boundary_token_count, 1)
        self.register_buffer("boundary_pos", boundary_pos, persistent=False)
        self.node_mlp = MLP(global_branch_size + 2, n_hidden, n_hidden, mlp_layers, act)
        self.to_q = nn.Linear(n_hidden, n_hidden, bias=False)
        self.to_k = nn.Linear(n_hidden, n_hidden, bias=False)
        self.to_v = nn.Linear(n_hidden, n_hidden, bias=False)
        self.edge_bias = nn.Sequential(
            nn.Linear(2, n_hidden),
            nn.GELU(),
            nn.Linear(n_hidden, n_head),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
        )

    def _local_mask(self, n_token: int, device: torch.device) -> torch.Tensor | None:
        if self.graph_k <= 0 or self.graph_k >= n_token - 1:
            return None
        idx = torch.arange(n_token, device=device)
        dist = (idx[:, None] - idx[None, :]).abs()
        return dist <= self.graph_k

    def forward(self, boundary_flux: torch.Tensor, global_branch: torch.Tensor) -> torch.Tensor:
        bsz, n_token, _ = boundary_flux.shape
        boundary_pos = self.boundary_pos[:, :n_token, :].expand(bsz, -1, -1)
        global_rep = global_branch.expand(-1, n_token, -1)
        node = self.node_mlp(torch.cat([boundary_flux, boundary_pos, global_rep], dim=-1))

        q = self.to_q(node).view(bsz, n_token, self.n_head, self.head_dim).transpose(1, 2)
        k = self.to_k(node).view(bsz, n_token, self.n_head, self.head_dim).transpose(1, 2)
        v = self.to_v(node).view(bsz, n_token, self.n_head, self.head_dim).transpose(1, 2)
        score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        pos_delta = (boundary_pos[:, :, None, :] - boundary_pos[:, None, :, :]).abs()
        flux_delta = (boundary_flux[:, :, None, :] - boundary_flux[:, None, :, :]).abs()
        edge = torch.cat([pos_delta, flux_delta], dim=-1)
        bias = self.edge_bias(edge).permute(0, 3, 1, 2)
        score = score + bias

        mask = self._local_mask(n_token, boundary_flux.device)
        if mask is not None:
            score = score.masked_fill(~mask.view(1, 1, n_token, n_token), torch.finfo(score.dtype).min)

        attn = F.softmax(score, dim=-1)
        graph_msg = (attn @ v).transpose(1, 2).contiguous().view(bsz, n_token, -1)
        return self.out(graph_msg)


class SpatialHeatGraphPropagation(nn.Module):
    def __init__(self, n_hidden: int, graph_k: int, dropout: float) -> None:
        super().__init__()
        self.graph_k = graph_k
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.message_proj = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        if self.graph_k <= 0 or x.shape[1] <= 1:
            return x
        n_token = x.shape[1]
        k_eff = min(self.graph_k, n_token - 1)
        dist = torch.cdist(pos, pos).clamp_min(1.0e-6)
        knn_dist, knn_idx = dist.topk(k_eff + 1, dim=-1, largest=False)
        knn_dist = knn_dist[:, :, 1:]
        knn_idx = knn_idx[:, :, 1:]

        gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1])
        neighbor = x.gather(1, knn_idx.reshape(x.shape[0], -1).unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        neighbor = neighbor.reshape(x.shape[0], n_token, k_eff, x.shape[-1])
        del gather_idx

        tau = self.log_tau.exp().clamp(0.05, 10.0)
        weights = F.softmax(-knn_dist / tau, dim=-1).unsqueeze(-1)
        diffusion = (weights * (neighbor - x.unsqueeze(2))).sum(dim=2)
        return torch.tanh(self.gate) * self.message_proj(diffusion)


class PNOTBlock(nn.Module):
    def __init__(
        self,
        n_hidden: int,
        n_head: int,
        n_experts: int,
        n_inner: int,
        space_dim: int,
        act: str,
        dropout: float,
        use_physics_attention: bool,
        physics_slice_num: int,
        use_heat_graph_propagation: bool,
        heat_graph_k: int,
    ) -> None:
        super().__init__()
        acts = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        activation = acts[act]
        hidden_inner = n_inner * n_hidden
        self.ln1 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)
        self.ln3 = nn.LayerNorm(n_hidden)
        self.ln4 = nn.LayerNorm(n_hidden)
        self.cross_attn = LinearAttention(n_hidden, n_head, dropout)
        if use_physics_attention:
            self.self_attn = PhysicsStateAttention(n_hidden, n_head, physics_slice_num, dropout)
        else:
            self.self_attn = LinearAttention(n_hidden, n_head, dropout)
        self.heat_graph = SpatialHeatGraphPropagation(n_hidden, heat_graph_k, dropout) if use_heat_graph_propagation else None
        self.ln_heat = nn.LayerNorm(n_hidden)
        self.drop = nn.Dropout(dropout)
        self.n_experts = n_experts
        self.moe1 = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, hidden_inner), activation(), nn.Linear(hidden_inner, n_hidden)) for _ in range(n_experts)]
        )
        self.moe2 = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, hidden_inner), activation(), nn.Linear(hidden_inner, n_hidden)) for _ in range(n_experts)]
        )
        self.gatenet = nn.Sequential(
            nn.Linear(space_dim, hidden_inner),
            activation(),
            nn.Linear(hidden_inner, hidden_inner),
            activation(),
            nn.Linear(hidden_inner, n_experts),
        )

    def _moe(self, experts: nn.ModuleList, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        gate = F.softmax(self.gatenet(pos), dim=-1).unsqueeze(2)
        expert_values = torch.stack([expert(x) for expert in experts], dim=-1)
        return (gate * expert_values).sum(dim=-1)

    def forward(self, x: torch.Tensor, branch: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.cross_attn(self.ln1(x), self.ln2(branch)))
        x = x + self.ln3(self._moe(self.moe1, x, pos))
        x = x + self.drop(self.self_attn(self.ln4(x)))
        if self.heat_graph is not None:
            x = x + self.drop(self.heat_graph(self.ln_heat(x), pos))
        x = x + self._moe(self.moe2, x, pos)
        return x


class HeatPNOT(nn.Module):
    def __init__(
        self,
        trunk_size: int,
        branch_size: int,
        output_size: int,
        n_layers: int,
        n_hidden: int,
        n_head: int,
        n_experts: int,
        n_inner: int,
        mlp_layers: int,
        attn_type: str,
        act: str,
        ffn_dropout: float,
        attn_dropout: float,
        horiz_fourier_dim: int,
        space_dim: int,
        use_boundary_tokens: bool = False,
        boundary_token_count: int = 53,
        global_branch_size: int = 5,
        use_physics_attention: bool = False,
        physics_slice_num: int = 64,
        use_boundary_relation_graph: bool = False,
        boundary_graph_k: int = 0,
        use_heat_graph_propagation: bool = False,
        heat_graph_k: int = 8,
    ) -> None:
        super().__init__()
        if attn_type != "linear":
            raise ValueError("HeatPNOT currently supports attn_type='linear'")
        if horiz_fourier_dim != 0:
            raise ValueError("HeatPNOT currently expects horiz_fourier_dim=0")
        if use_boundary_tokens and branch_size != global_branch_size + boundary_token_count:
            raise ValueError("branch_size must equal global_branch_size + boundary_token_count")
        self.space_dim = space_dim
        self.branch_size = branch_size
        self.use_boundary_tokens = use_boundary_tokens
        self.use_boundary_relation_graph = use_boundary_relation_graph
        self.boundary_token_count = boundary_token_count
        self.global_branch_size = global_branch_size
        self.trunk_mlp = MLP(trunk_size, n_hidden, n_hidden, mlp_layers, act)
        if use_boundary_tokens:
            self.global_branch_mlp = MLP(global_branch_size, n_hidden, n_hidden, mlp_layers, act)
            self.boundary_flux_mlp = MLP(1, n_hidden, n_hidden, mlp_layers, act)
            if use_boundary_relation_graph:
                self.boundary_graph_gate = nn.Parameter(torch.tensor(0.0))
                self.boundary_graph_encoder = BoundaryHeatFluxGraphEncoder(
                    global_branch_size,
                    boundary_token_count,
                    n_hidden,
                    n_head,
                    mlp_layers,
                    act,
                    boundary_graph_k,
                    max(ffn_dropout, attn_dropout),
                )
            self.boundary_pos_embed = nn.Parameter(torch.zeros(1, boundary_token_count, n_hidden))
            nn.init.trunc_normal_(self.boundary_pos_embed, std=0.02)
        else:
            self.branch_mlp = MLP(branch_size, n_hidden, n_hidden, mlp_layers, act)
        self.blocks = nn.ModuleList(
            [
                PNOTBlock(
                    n_hidden,
                    n_head,
                    n_experts,
                    n_inner,
                    space_dim,
                    act,
                    max(ffn_dropout, attn_dropout),
                    use_physics_attention,
                    physics_slice_num,
                    use_heat_graph_propagation,
                    heat_graph_k,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_mlp = MLP(n_hidden, n_hidden, output_size, mlp_layers, act)

    def encode_branch(self, branch: torch.Tensor) -> torch.Tensor:
        if not self.use_boundary_tokens:
            return self.branch_mlp(branch)

        global_branch = branch[:, :, : self.global_branch_size]
        boundary_flux = branch[:, :, self.global_branch_size :]
        global_token = self.global_branch_mlp(global_branch)
        boundary_tokens = self.boundary_flux_mlp(boundary_flux.transpose(1, 2))
        if self.use_boundary_relation_graph:
            graph_tokens = self.boundary_graph_encoder(boundary_flux.transpose(1, 2), global_branch)
            boundary_tokens = boundary_tokens + torch.tanh(self.boundary_graph_gate) * graph_tokens
        boundary_tokens = boundary_tokens + self.boundary_pos_embed
        return torch.cat([global_token, boundary_tokens], dim=1)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        trunk = model_input[:, :, :3]
        pos = trunk[:, :, : self.space_dim]
        branch = model_input[:, :1, 3:]
        x = self.trunk_mlp(trunk)
        z = self.encode_branch(branch)
        for block in self.blocks:
            x = block(x, z, pos)
        return self.out_mlp(x).squeeze(-1)
