import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATv2Conv
from typing import Tuple

class MLP(nn.Module):
    """Basic MLP with dropout and GELU activation."""

    def __init__(
        self,
        hidden_dim: int,
        add_linear_last: bool,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(num_layers):
            self.layers.append(nn.LazyLinear(hidden_dim))
            self.layers.append(nn.ELU())
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(p=dropout_rate))
        if add_linear_last:
            self.layers.append(nn.LazyLinear(hidden_dim))

    def forward(self, x):
        output = self.layers(x)
        return output

class MixtureEncoder(nn.Module):
    """
    Mixture-level GAT (Graph Attention Network v2).

    Changes
    -------
    1. GraphConv  →  two GATv2Conv layers (GATv2 has stronger attention
       expressiveness than GATv1 and can distinguish different neighbor pairs).
    2. Multi-head attention: the first layer uses num_heads independent heads
       with concat output; the second layer uses 1 head for the final aggregation
       with mean output.
    3. Dropout + residual projection between layers (linear alignment if
       dimensions differ).
    4. The other interfaces (build_mixture_graph / emb / forward / rho /
       _graph_cache) remain fully compatible with the original GCN version,
       so external callers need no changes.

    Input tensor x: (batch_size, 43, mol_dim, molecule_num)
      - 43:           max molecules per mixture, padded with -999
      - mol_dim:      DMPNN per-molecule embedding dimension (e.g. 512)
      - molecule_num: 1 for label task, 2 for similarity task

    Output:
      - 'label':      (batch_size, 51)
      - 'similarity': (batch_size,)
    """

    def __init__(
        self,
        mol_dim: int,
        hidden_dim: int,
        output_type: str,
        num_heads: int = 4,
        dropout: float = 0.1, # 0.1
    ):
        """
        Args:
            mol_dim:     input node feature dimension (DMPNN embedding dim, e.g. 512)
            hidden_dim:  GAT hidden dimension; also the input dimension of rho
            output_type: 'label' or 'similarity'
            num_heads:   number of attention heads in the first GAT layer (recommended 4 or 8)
            dropout:     dropout probability for attention weights and features
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.output_type = output_type
        head_dim = hidden_dim // num_heads  # per-head feature dimension

        # ── Layer 1 GATv2: mol_dim → num_heads × head_dim (concat) ──
        self.gat1 = GATv2Conv(
            in_feats=mol_dim,
            out_feats=head_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=F.elu,
            allow_zero_in_degree=True,   # support isolated nodes (single-component mixtures)
        )

        # Layer-1 residual projection (mol_dim → hidden_dim)
        self.res_proj1 = nn.Linear(mol_dim, hidden_dim, bias=False)
        self.res_proj2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # ── Layer 2 GATv2: hidden_dim → hidden_dim (single-head mean) ──
        self.gat2 = GATv2Conv(
            in_feats=hidden_dim,
            out_feats=head_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=F.elu,
            allow_zero_in_degree=True,   # support isolated nodes (single-component mixtures)
        )

        self.gat3 = GATv2Conv(
            in_feats=hidden_dim,
            out_feats=hidden_dim,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=None,    #None         # activation applied after residual addition
            allow_zero_in_degree=True,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)  # self-add
        self.act   = nn.ELU()
        self.drop1  = nn.Dropout(dropout)
        self.drop2  = nn.Dropout(dropout)

        # ── Output head: input dim is hidden_dim * 4 (after PNA concat) ──
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ELU(),
            nn.Linear(hidden_dim // 4, 51),
        )

        # Graph topology cache (same as the original)
        self._graph_cache: dict = {}

        self.ffn = MLP(
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
            add_linear_last=False,
        )
    # ------------------------------------------------------------------
    # build_mixture_graph: identical to the original, no changes
    # ------------------------------------------------------------------
    def build_mixture_graph(
        self, x_mix: torch.Tensor
    ) -> Tuple[dgl.DGLGraph, torch.Tensor]:
        """
        Construct a batched mixture graph from a padded embedding tensor.

        Args:
            x_mix: (batch_size, 43, mol_dim)

        Returns:
            batched_graph: DGL batched graph
            mol_embs:      (total_valid_mols, mol_dim)
        """
        device = x_mix.device
        valid_mask = ~(x_mix == -999).all(dim=-1)   # (batch_size, 43)
        mol_embs   = x_mix[valid_mask]               # (total_valid_mols, mol_dim)

        cache_key = valid_mask.cpu().numpy().tobytes()
        if cache_key not in self._graph_cache:
            graphs = []
            for i in range(x_mix.shape[0]):
                n = int(valid_mask[i].sum().item())
                idx = torch.arange(n)
                src = idx.repeat_interleave(n)        # complete graph + self-loops
                dst = idx.repeat(n)
                graphs.append(dgl.graph((src, dst)))
            self._graph_cache[cache_key] = dgl.batch(graphs)

        batched_graph = self._graph_cache[cache_key].to(device)
        return batched_graph, mol_embs

    # ------------------------------------------------------------------
    # emb: GCN replaced by two GATv2 layers + residual + LayerNorm
    # ------------------------------------------------------------------
    def emb(
        self, mixture_graph: dgl.DGLGraph, mol_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Two-layer GATv2 message passing + mean pooling.

        Args:
            mixture_graph: batched DGL graph
            mol_embs:      (total_valid_mols, mol_dim)

        Returns:
            (batch_size, hidden_dim)
        """
        h = mol_embs                                    # (V, mol_dim)

        # ── Layer 1 ──────────────────────────────────────────────────
        # GATv2Conv output: (V, num_heads, head_dim)
        h1 = self.gat1(mixture_graph, h)                # (V, num_heads, head_dim)
        h1 = h1.flatten(1)                              # (V, hidden_dim)  [concat heads]
        h1 = h1 + self.res_proj1(h)                     # residual connection
        h1 = self.norm1(h1)                             # LayerNorm
        h1 = self.drop1(h1)

        # ── Layer 2 ──────────────────────────────────────────────────
        # GATv2Conv output: (V, 1, hidden_dim)
        h2 = self.gat2(mixture_graph, h1)               # (V, 1, hidden_dim)
        h2 = h2.flatten(1)                              # (V, hidden_dim)  [single head]
        h2 = h2 + self.res_proj2(h1)                          # residual + ELU
        h2 = self.norm2(h2)
        h2 = self.drop2(h2)                              # self-add
        # self
        # ── Layer 3 ──────────────────────────────────────────────────
        # GATv2Conv output: (V, 1, hidden_dim)
        h3 = self.gat3(mixture_graph, h2)               # (V, 1, hidden_dim)
        h3 = h3.squeeze(1)                              # (V, hidden_dim)  [single head]
        h3 = self.act(h3 + h2)                          # residual + ELU
        h3 = self.norm3(h3)

        # ── PNA pooling over nodes (mean / std / min / max) ──────────
        mixture_graph.ndata['h'] = h3
        return self.ffn(self.pna_readout(mixture_graph, 'h'))      # (batch_size, hidden_dim * 4)
        # return dgl.mean_nodes(mixture_graph, 'h')        # (batch_size, hidden_dim)

    # ------------------------------------------------------------------
    # pna_readout: replaces dgl.mean_nodes, concatenates mean / std / min / max
    # ------------------------------------------------------------------
    def pna_readout(
        self, mixture_graph: dgl.DGLGraph, feat_key: str = 'h'
    ) -> torch.Tensor:
        """
        PNA-style graph-level readout.

        For each subgraph in the batched graph, apply four aggregations on
        the node features and concatenate them:
            mean | std | min | max  →  (batch_size, hidden_dim * 4)
        """
        h_mean = dgl.readout_nodes(mixture_graph, feat_key, op='mean')  # (B, D)
        h_max  = dgl.readout_nodes(mixture_graph, feat_key, op='max')   # (B, D)
        h_min  = dgl.readout_nodes(mixture_graph, feat_key, op='min')   # (B, D)

        # std = sqrt( E[x^2] - (E[x])^2 ), computed via two readouts
        mixture_graph.ndata['_h_sq'] = mixture_graph.ndata[feat_key] ** 2
        h_mean_sq = dgl.readout_nodes(mixture_graph, '_h_sq', op='mean')  # (B, D)
        h_std = torch.sqrt(torch.clamp(h_mean_sq - h_mean ** 2, min=1e-8))

        return torch.cat([h_mean, h_std, h_min, h_max], dim=-1)  # (B, D*4)

    # ------------------------------------------------------------------
    # forward: identical to the original, no changes
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 43, mol_dim, molecule_num)

        Returns:
            label task:      (batch_size, 51)
            similarity task: (batch_size,)
        """
        embeddings = []
        for x_mix in torch.unbind(x, dim=-1):            # (batch_size, 43, mol_dim)
            batched_graph, mol_embs = self.build_mixture_graph(x_mix)
            embeddings.append(self.emb(batched_graph, mol_embs))
        final_emb = torch.stack(embeddings, dim=-1)       # (batch_size, hidden_dim*4, molecule_num)

        if self.output_type == 'label':
            return self.rho(final_emb[..., 0])            # (batch_size, 51)

        elif self.output_type == 'similarity':
            emb1 = final_emb[..., 0]
            emb2 = final_emb[..., 1]
            # score = torch.abs(
            #     emb1 / emb1.norm(p=1, dim=1, keepdim=True) -
            #     emb2 / emb2.norm(p=1, dim=1, keepdim=True)
            # ).sum(dim=1)

            score = torch.abs(emb1  -  emb2).mean(dim=1)
            # score = 1 - F.cosine_similarity(emb1,emb2,dim=1)
            # score = (score - score.mean()) / score.std()
            return score                                   # (batch_size,)

        else:
            raise ValueError(
                f"output_type must be 'label' or 'similarity', got '{self.output_type}'"
            )