# -*- coding: utf-8 -*-
"""Hydro-MAST — Mask-aware Advective Graph + GRU (가성비 대안 AI)"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hydro_mast_data import N_HRFC, TARGET_NODE, build_edge_index


class AdvectiveGraphLayer(nn.Module):
    """간선별 학습 지연(이산 bin)으로 상류 상태를 하류로 전달"""

    def __init__(
        self,
        hidden: int,
        n_delay_bins: int = 5,
        edge_src: np.ndarray | None = None,
        edge_dst: np.ndarray | None = None,
    ):
        super().__init__()
        if edge_src is None or edge_dst is None:
            edge_src, edge_dst = build_edge_index()
        self.register_buffer("edge_src", torch.from_numpy(edge_src))
        self.register_buffer("edge_dst", torch.from_numpy(edge_dst))
        self.n_edges = len(edge_src)
        self.n_delay_bins = n_delay_bins
        self.msg = nn.Linear(hidden, hidden, bias=False)
        self.delay_logits = nn.Parameter(torch.zeros(self.n_edges, n_delay_bins))

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: (B, T, N, H)"""
        b, t, _, hid = h_seq.shape
        k = self.n_delay_bins
        weights = torch.softmax(self.delay_logits, dim=-1)
        out = h_seq.clone()
        for e in range(self.n_edges):
            u = int(self.edge_src[e])
            v = int(self.edge_dst[e])
            hu = h_seq[:, :, u, :]
            delayed = [hu]
            for lag in range(1, k):
                pad = torch.zeros(b, lag, hid, device=h_seq.device, dtype=h_seq.dtype)
                delayed.append(torch.cat([pad, hu[:, :-lag, :]], dim=1))
            stack = torch.stack(delayed, dim=2)
            w = weights[e].view(1, 1, k, 1)
            mixed = (stack * w).sum(dim=2)
            out[:, :, v, :] = out[:, :, v, :] + self.msg(mixed)
        return out


class HydroMAST(nn.Module):
    def __init__(
        self,
        hrfc_channels: int = 2,
        dam_channels: int = 4,
        time_dim: int = 4,
        hidden: int = 64,
        n_horizons: int = 4,
        n_delay_bins: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.hrfc_enc = nn.Linear(hrfc_channels, hidden)
        self.dam_enc = nn.Linear(dam_channels, hidden)
        self.time_proj = nn.Linear(time_dim, hidden)
        self.graph = AdvectiveGraphLayer(hidden, n_delay_bins=n_delay_bins)
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.horizon_emb = nn.Embedding(n_horizons, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_horizons)])
        self.drop = nn.Dropout(dropout)

    def encode_nodes(self, nodes: torch.Tensor, n_dam_ch: int) -> torch.Tensor:
        """nodes: (B, T, N, C_pad) — HRFC 2ch + dam padded"""
        b, t, n, c = nodes.shape
        hrfc = nodes[:, :, :N_HRFC, :2]
        dam = nodes[:, :, N_HRFC : N_HRFC + 1, :n_dam_ch]
        h_h = torch.tanh(self.hrfc_enc(hrfc))
        h_d = torch.tanh(self.dam_enc(dam))
        h = torch.zeros(b, t, n, self.hidden, device=nodes.device, dtype=nodes.dtype)
        h[:, :, :N_HRFC, :] = h_h
        h[:, :, N_HRFC, :] = h_d.squeeze(2)
        return h

    def forward(
        self,
        nodes: torch.Tensor,
        time_g: torch.Tensor,
        n_dam_ch: int,
    ) -> torch.Tensor:
        """
        nodes: (B, T, N, C)
        time_g: (B, time_dim)
        -> (B, n_horizons)
        """
        h = self.encode_nodes(nodes, n_dam_ch)
        h = self.graph(h)
        h_pool = h.mean(dim=2)
        h_seq, _ = self.gru(h_pool)
        h_last = self.drop(h_seq[:, -1, :])
        t_emb = torch.tanh(self.time_proj(time_g))
        h_ctx = h_last + t_emb

        preds = []
        dev = nodes.device
        for i, head in enumerate(self.heads):
            idx = torch.full((h_ctx.size(0),), i, dtype=torch.long, device=dev)
            he = self.horizon_emb(idx)
            preds.append(head(torch.cat([h_ctx, he], dim=-1)))
        return torch.cat(preds, dim=1)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=delta)
