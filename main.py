import argparse
import copy
import math
import os
import os.path as osp
import random
import time
import urllib.request
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, LINKXDataset, Planetoid, WebKB, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import remove_self_loops, to_undirected


# -----------------------------
# Models
# -----------------------------

class GCNIIConv(nn.Module):
    def __init__(self, hidden_dim, variant=False, residual=False):
        super().__init__()
        self.variant = variant
        self.residual = residual
        in_dim = hidden_dim * 2 if variant else hidden_dim
        self.weight = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x, x0, adj, lamda, alpha, layer_id):
        theta = math.log(lamda / layer_id + 1.0)
        hi = torch.sparse.mm(adj, x)

        if self.variant:
            support = torch.cat([hi, x0], dim=1)
            r = (1.0 - alpha) * hi + alpha * x0
        else:
            support = (1.0 - alpha) * hi + alpha * x0
            r = support

        out = theta * torch.mm(support, self.weight) + (1.0 - theta) * r

        if self.residual:
            out = out + x

        return out


class GCNII(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout, lamda, alpha, variant):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNIIConv(hidden_dim, variant=variant) for _ in range(layers)
        ])
        self.fcs = nn.ModuleList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        ])
        self.dropout = dropout
        self.lamda = lamda
        self.alpha = alpha
        self.act = nn.ReLU()

        # Keep the original GCNII optimizer grouping.
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        h0 = self.act(self.fcs[0](x))
        h = h0

        for idx, conv in enumerate(self.convs):
            h = F.dropout(h, self.dropout, training=self.training)
            h = self.act(conv(h, h0, adj, self.lamda, self.alpha, idx + 1))

        h = F.dropout(h, self.dropout, training=self.training)
        return self.fcs[-1](h)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout, use_bn=False):
        super().__init__()
        modules = []

        if layers <= 1:
            modules.append(nn.Linear(in_dim, out_dim))
        else:
            modules.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                modules.append(nn.BatchNorm1d(hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))

            for _ in range(layers - 2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    modules.append(nn.BatchNorm1d(hidden_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))

            modules.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MLPNet(nn.Module):
    """MLP baseline with the same forward signature as graph models."""

    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout, use_bn=False):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, layers, dropout, use_bn=use_bn)

    def forward(self, x, adj=None):
        return self.mlp(x)


class APPNPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, alpha, dropout, use_bn=False):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout, use_bn=use_bn)
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x, adj):
        z0 = self.mlp(F.dropout(x, self.dropout, training=self.training))
        z = z0

        for _ in range(self.propagation_steps):
            z = (1.0 - self.alpha) * torch.sparse.mm(adj, z) + self.alpha * z0

        return z


class DAGNNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, dropout, use_bn=False):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout, use_bn=use_bn)
        self.att = nn.Linear(out_dim, 1, bias=False)
        self.propagation_steps = propagation_steps
        self.dropout = dropout

    def forward(self, x, adj):
        z = self.mlp(F.dropout(x, self.dropout, training=self.training))
        outs = [z]

        for _ in range(self.propagation_steps):
            z = torch.sparse.mm(adj, z)
            outs.append(z)

        stack = torch.stack(outs, dim=1)
        weight = torch.softmax(self.att(stack).squeeze(-1), dim=1)
        return (weight.unsqueeze(-1) * stack).sum(dim=1)


class GPRNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, alpha, dropout, init, use_bn=False):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout, use_bn=use_bn)
        self.propagation_steps = propagation_steps
        self.dropout = dropout

        temp = self.init_weights(propagation_steps, alpha, init)
        self.temp = nn.Parameter(temp)

    def init_weights(self, steps, alpha, init):
        if init == 'ppr':
            vals = [alpha * (1.0 - alpha) ** k for k in range(steps)]
            vals.append((1.0 - alpha) ** steps)
            return torch.tensor(vals, dtype=torch.float)

        if init == 'nppr':
            vals = [(alpha ** k) for k in range(steps + 1)]
            vals = torch.tensor(vals, dtype=torch.float)
            return vals / (vals.abs().sum() + 1e-12)

        if init == 'alt':
            # Alternating signed initialization for heterophilic graphs.
            vals = [((-alpha) ** k) for k in range(steps + 1)]
            vals = torch.tensor(vals, dtype=torch.float)
            return vals / (vals.abs().sum() + 1e-12)

        if init == 'random':
            # Matches the commonly used GPR-GNN random coefficient start.
            bound = math.sqrt(3.0 / (steps + 1))
            vals = torch.empty(steps + 1, dtype=torch.float).uniform_(-bound, bound)
            return vals / (vals.abs().sum() + 1e-12)

        # sgc
        vals = torch.zeros(steps + 1, dtype=torch.float)
        vals[-1] = 1.0
        return vals

    def forward(self, x, adj):
        z = self.mlp(F.dropout(x, self.dropout, training=self.training))
        out = self.temp[0] * z
        h = z

        for k in range(1, self.propagation_steps + 1):
            h = torch.sparse.mm(adj, h)
            out = out + self.temp[k] * h

        return out


class H2GCNLite(nn.Module):
    """
    Lightweight H2GCN-style model.

    It keeps ego representation h0 and concatenates propagated representations.
    For this model, use adjacency without self-loops to avoid mixing ego and
    neighbor channels too early.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, hops, dropout, use_bn=False):
        super().__init__()
        self.hops = hops
        self.dropout = dropout

        self.lin = nn.Linear(in_dim, hidden_dim)
        self.pred = MLP(hidden_dim * (hops + 1), hidden_dim, out_dim, 2, dropout, use_bn=use_bn)

    def forward(self, x, adj):
        h0 = F.dropout(x, self.dropout, training=self.training)
        h0 = F.relu(self.lin(h0))

        outs = [h0]
        h = h0

        for _ in range(self.hops):
            h = torch.sparse.mm(adj, h)
            outs.append(h)

        h = torch.cat(outs, dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        return self.pred(h)


class LINKXNet(nn.Module):
    """
    LINKX-style model close to the official implementation.

    a = MLP_A(A), x = MLP_X(X), h = ReLU(W[a, x] + a + x), y = MLP_final(h).
    The adjacency branch is implemented as sparse A @ W to avoid materializing
    a dense N x N input matrix.
    """

    def __init__(
        self,
        in_dim,
        num_nodes,
        hidden_dim,
        out_dim,
        dropout,
        use_bn=False,
        skip=True,
        init_layers_x=1,
        final_layers=2,
        inner_activation=False,
        inner_dropout=False,
    ):
        super().__init__()
        self.dropout = dropout
        self.skip = skip
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

        self.adj_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.adj_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Official LINKX uses zero dropout in the initial A/X embedding MLPs and
        # applies dropout in the final MLP.  Keeping this separation matters on
        # the large non-homophily benchmarks.
        self.x_mlp = MLP(
            in_dim,
            hidden_dim,
            hidden_dim,
            init_layers_x,
            dropout=0.0,
            use_bn=use_bn,
        )
        self.merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final_mlp = MLP(
            hidden_dim,
            hidden_dim,
            out_dim,
            final_layers,
            dropout=dropout,
            use_bn=use_bn,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.adj_weight)
        nn.init.zeros_(self.adj_bias)
        self.merge.reset_parameters()

    def forward(self, x, adj):
        # For LINKX, pass raw adjacency: no normalization and no self-loops.
        a_emb = torch.sparse.mm(adj, self.adj_weight) + self.adj_bias
        a_emb = F.relu(a_emb)

        x_emb = self.x_mlp(x)

        h = self.merge(torch.cat([a_emb, x_emb], dim=1))
        if self.inner_dropout:
            h = F.dropout(h, self.dropout, training=self.training)
        if self.inner_activation:
            h = F.relu(h)

        if self.skip:
            h = F.relu(h + a_emb + x_emb)
        else:
            h = F.relu(h)

        return self.final_mlp(h)


class GCNNet(nn.Module):
    """Small full-batch GCN baseline using the same sparse adjacency path."""

    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout, use_bn=False):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        if layers <= 1:
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(nn.Linear(in_dim, hidden_dim))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x, adj):
        if len(self.lins) == 1:
            return self.lins[0](torch.sparse.mm(adj, x))

        h = x
        for i, lin in enumerate(self.lins[:-1]):
            h = torch.sparse.mm(adj, h)
            h = lin(h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)

        h = torch.sparse.mm(adj, h)
        return self.lins[-1](h)


class MixHopLayer(nn.Module):
    """MixHop layer: concatenate learned projections from A^0, A^1, ..., A^K."""

    def __init__(self, in_dim, out_dim, hops):
        super().__init__()
        self.hops = hops
        self.lins = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(hops + 1)
        ])

    def forward(self, x, adj):
        outs = [self.lins[0](x)]
        for hop in range(1, self.hops + 1):
            h = self.lins[hop](x)
            for _ in range(hop):
                h = torch.sparse.mm(adj, h)
            outs.append(h)
        return torch.cat(outs, dim=1)


class MixHopNet(nn.Module):
    """Scalable MixHop baseline used as a strong non-homophily comparator."""

    def __init__(self, in_dim, hidden_dim, out_dim, layers, hops, dropout, use_bn=False):
        super().__init__()
        self.dropout = dropout
        self.use_bn = use_bn
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if layers <= 1:
            self.convs.append(MixHopLayer(in_dim, out_dim, hops))
            self.final_project = nn.Linear(out_dim * (hops + 1), out_dim)
        else:
            self.convs.append(MixHopLayer(in_dim, hidden_dim, hops))
            if use_bn:
                self.bns.append(nn.BatchNorm1d(hidden_dim * (hops + 1)))
            for _ in range(layers - 2):
                self.convs.append(MixHopLayer(hidden_dim * (hops + 1), hidden_dim, hops))
                if use_bn:
                    self.bns.append(nn.BatchNorm1d(hidden_dim * (hops + 1)))
            self.convs.append(MixHopLayer(hidden_dim * (hops + 1), out_dim, hops))
            self.final_project = nn.Linear(out_dim * (hops + 1), out_dim)

    def forward(self, x, adj):
        h = x
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(h, adj)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        h = self.convs[-1](h, adj)
        return self.final_project(h)


class ACMGCNLayer(nn.Module):
    """
    Adaptive Channel Mixing layer, simplified for this standalone script.

    It mixes three channels node-wise: low-pass aggregation, high-pass
    diversification, and identity/MLP.  This is a practical ACM-GCN-lite option
    for heterophily; it does not enable the optional structure-info channel.
    """

    def __init__(self, in_dim, out_dim, use_layer_norm=True, variant=False):
        super().__init__()
        self.variant = variant
        self.use_layer_norm = use_layer_norm

        self.weight_low = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_high = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_mlp = nn.Parameter(torch.empty(in_dim, out_dim))

        self.att_vec_low = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec_high = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec_mlp = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec = nn.Parameter(torch.empty(3, 3))

        if use_layer_norm:
            self.norm_low = nn.LayerNorm(out_dim)
            self.norm_high = nn.LayerNorm(out_dim)
            self.norm_mlp = nn.LayerNorm(out_dim)
        else:
            self.norm_low = self.norm_high = self.norm_mlp = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight_mlp.size(1))
        att_bound = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        mix_bound = 1.0 / math.sqrt(self.att_vec.size(1))
        for weight in [self.weight_low, self.weight_high, self.weight_mlp]:
            nn.init.uniform_(weight, -bound, bound)
        for weight in [self.att_vec_low, self.att_vec_high, self.att_vec_mlp]:
            nn.init.uniform_(weight, -att_bound, att_bound)
        nn.init.uniform_(self.att_vec, -mix_bound, mix_bound)
        for norm in [self.norm_low, self.norm_high, self.norm_mlp]:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def _attention(self, low, high, mlp):
        logits = torch.cat([
            torch.mm(self.norm_low(low), self.att_vec_low),
            torch.mm(self.norm_high(high), self.att_vec_high),
            torch.mm(self.norm_mlp(mlp), self.att_vec_mlp),
        ], dim=1)
        logits = torch.mm(torch.sigmoid(logits), self.att_vec) / 3.0
        att = torch.softmax(logits, dim=1)
        return att[:, 0:1], att[:, 1:2], att[:, 2:3]

    def forward(self, x, adj_low, adj_high):
        if self.variant:
            low = torch.sparse.mm(adj_low, F.relu(torch.mm(x, self.weight_low)))
            high = torch.sparse.mm(adj_high, F.relu(torch.mm(x, self.weight_high)))
            mlp = F.relu(torch.mm(x, self.weight_mlp))
        else:
            low = F.relu(torch.sparse.mm(adj_low, torch.mm(x, self.weight_low)))
            high = F.relu(torch.sparse.mm(adj_high, torch.mm(x, self.weight_high)))
            mlp = F.relu(torch.mm(x, self.weight_mlp))

        att_low, att_high, att_mlp = self._attention(low, high, mlp)
        return 3.0 * (att_low * low + att_high * high + att_mlp * mlp)


class ACMGCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout, use_bn=True, variant=False, plus_plus=True):
        super().__init__()
        if layers != 2:
            # The official large-scale ACM-Geometric branch uses two layers.
            layers = 2
        self.dropout = dropout
        self.plus_plus = plus_plus
        self.conv1 = ACMGCNLayer(in_dim, hidden_dim, use_layer_norm=use_bn, variant=variant)
        self.conv2 = ACMGCNLayer(hidden_dim, out_dim, use_layer_norm=use_bn, variant=variant)
        if plus_plus:
            self.x_skip = MLP(in_dim, hidden_dim, hidden_dim, 1, dropout=0.0, use_bn=False)
        else:
            self.x_skip = None

    def forward(self, x, adj):
        if isinstance(adj, tuple):
            adj_low, adj_high = adj
        else:
            adj_low, adj_high = adj, sparse_identity_minus(adj)

        x_drop = F.dropout(x, self.dropout, training=self.training)
        h = self.conv1(x_drop, adj_low, adj_high)
        h = F.dropout(F.relu(h), self.dropout, training=self.training)
        if self.x_skip is not None:
            h = h + F.dropout(F.relu(self.x_skip(x)), self.dropout, training=self.training)
        return self.conv2(h, adj_low, adj_high)


# -----------------------------
# Utilities
# -----------------------------

SNAP_PATENTS_GDRIVE_ID = '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia'
ARXIV_YEAR_SPLIT_URL = (
    'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/'
    'data/splits/arxiv-year-splits.npy'
)
SNAP_PATENTS_SPLIT_GDRIVE_ID = '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N'



class SingleGraphDataset:
    """Small adapter so external single-graph loaders look like PyG datasets."""

    def __init__(self, name, data, num_classes):
        self.name = name
        self.data = data
        self.num_classes = int(num_classes)
        self.num_node_features = int(data.num_node_features)

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('single graph dataset only contains graph 0')
        return self.data

    def __len__(self):
        return 1

    def __repr__(self):
        return f'{self.name}(1)'


def is_hetero_dataset(data_id):
    return data_id >= 3


def is_large_hetero_dataset(data_id):
    return data_id >= 9



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_mask(mask, split_id):
    if mask.dim() == 2:
        return mask[:, split_id % mask.size(1)].bool()
    return mask.bool()


def random_balanced_split(data, class_count, train_per_class, num_val, num_test, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    used = torch.zeros(data.num_nodes, dtype=torch.bool)

    for c in range(class_count):
        idx = (data.y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        chosen = idx[:min(train_per_class, idx.numel())]
        train_mask[chosen] = True
        used[chosen] = True

    remaining = (~used).nonzero(as_tuple=False).view(-1)
    remaining = remaining[torch.randperm(remaining.numel(), generator=generator)]

    val_n = min(num_val, remaining.numel())
    test_n = min(num_test, max(0, remaining.numel() - val_n))

    val_mask[remaining[:val_n]] = True
    test_mask[remaining[val_n:val_n + test_n]] = True

    if test_n == 0:
        test_mask[remaining[val_n:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def random_prop_split(data, train_prop, val_prop, seed, ignore_negative=True):
    """Random split used by the large heterophily benchmark datasets."""
    generator = torch.Generator()
    generator.manual_seed(seed)

    y = data.y.view(-1)
    if ignore_negative:
        idx = (y >= 0).nonzero(as_tuple=False).view(-1)
    else:
        idx = torch.arange(data.num_nodes)

    idx = idx[torch.randperm(idx.numel(), generator=generator)]
    train_n = int(idx.numel() * train_prop)
    val_n = int(idx.numel() * val_prop)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[idx[:train_n]] = True
    val_mask[idx[train_n:train_n + val_n]] = True
    test_mask[idx[train_n + val_n:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def _indices_to_mask(index, num_nodes):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    index = torch.as_tensor(index, dtype=torch.long).view(-1)
    index = index[(index >= 0) & (index < num_nodes)]
    mask[index] = True
    return mask


def set_masks_from_split(data, split):
    """Apply a LINKX benchmark split dict to a PyG Data object."""
    if isinstance(split, np.ndarray) and split.shape == ():
        split = split.item()
    if not isinstance(split, dict):
        raise TypeError('fixed split must be a dictionary of index arrays')

    train_key = 'train' if 'train' in split else 'train_idx'
    valid_key = 'valid' if 'valid' in split else ('val' if 'val' in split else 'valid_idx')
    test_key = 'test' if 'test' in split else 'test_idx'

    data.train_mask = _indices_to_mask(split[train_key], data.num_nodes)
    data.val_mask = _indices_to_mask(split[valid_key], data.num_nodes)
    data.test_mask = _indices_to_mask(split[test_key], data.num_nodes)
    return data


def _download_fixed_large_split(data_id, args):
    split_dir = osp.join(args.root, 'LINKX', 'fixed_splits')
    os.makedirs(split_dir, exist_ok=True)

    if data_id == 10:
        path = osp.join(split_dir, 'arxiv-year-splits.npy')
        if not osp.exists(path):
            urllib.request.urlretrieve(ARXIV_YEAR_SPLIT_URL, path)
        return path

    if data_id == 11:
        path = osp.join(split_dir, 'snap-patents-splits.npy')
        if not osp.exists(path):
            try:
                import gdown
            except ImportError as exc:
                raise ImportError(
                    'Fixed snap-patents splits require gdown. Install it with '
                    '`pip install gdown`, or pass --random-large-splits to use '
                    'the local random split fallback.'
                ) from exc
            gdown.download(
                id=SNAP_PATENTS_SPLIT_GDRIVE_ID,
                output=path,
                quiet=False
            )
        return path

    return None


def maybe_apply_fixed_large_split(data, data_id, args):
    """Use benchmark fixed splits for arXiv-year/snap-patents when available."""
    if not getattr(args, 'use_fixed_large_splits', False):
        return False
    if data_id not in [10, 11]:
        return False

    try:
        path = _download_fixed_large_split(data_id, args)
        splits = np.load(path, allow_pickle=True)
        if isinstance(splits, np.ndarray) and splits.shape == ():
            splits = [splits.item()]
        split = splits[args.split_id % len(splits)]
        set_masks_from_split(data, split)
        return True
    except Exception as exc:
        warnings.warn(
            f'Could not apply fixed benchmark split for data={data_id}; '
            f'falling back to the existing random split. Reason: {exc}'
        )
        return False


def sanitize_labels_and_masks(data, class_count):
    y = data.y.view(-1).long()
    valid = (y >= 0) & (y < class_count)

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, mask_name, None)
        if mask is not None:
            setattr(data, mask_name, mask.bool() & valid)

    data.y = torch.where(valid, y, torch.zeros_like(y))
    return data


def even_quantile_labels(vals, nclasses):
    """Convert continuous years to balanced ordinal classes by quantile."""
    vals = np.asarray(vals).reshape(-1)
    labels = -1 * np.ones(vals.shape[0], dtype=np.int64)
    lower = -np.inf

    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        labels[(vals >= lower) & (vals < upper)] = k
        lower = upper

    labels[vals >= lower] = nclasses - 1
    return labels


def make_large_dataset(name, data, class_count, args, transform):
    if transform is not None:
        data = transform(data)

    data.y = data.y.view(-1).long()
    data = random_prop_split(
        data,
        train_prop=args.train_prop,
        val_prop=args.val_prop,
        seed=args.seed,
        ignore_negative=True
    )
    data = sanitize_labels_and_masks(data, class_count)

    return SingleGraphDataset(name, data, class_count)


def load_arxiv_year_dataset(args, transform):
    try:
        from ogb.nodeproppred import NodePropPredDataset
    except ImportError as exc:
        raise ImportError(
            'arXiv-year requires OGB. Install it with `pip install ogb`.'
        ) from exc

    ogb_dataset = NodePropPredDataset(
        name='ogbn-arxiv',
        root=osp.join(args.root, 'ogb')
    )
    graph, _ = ogb_dataset[0]

    x = torch.as_tensor(graph['node_feat'], dtype=torch.float)
    edge_index = torch.as_tensor(graph['edge_index'], dtype=torch.long)
    years = np.asarray(graph['node_year']).reshape(-1)
    y = torch.as_tensor(
        even_quantile_labels(years, args.year_classes),
        dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.size(0))
    return make_large_dataset('arXiv-year', data, args.year_classes, args, transform)


def load_snap_patents_dataset(args, transform):
    try:
        import scipy.io
    except ImportError as exc:
        raise ImportError(
            'snap-patents requires scipy. Install it with `pip install scipy`.'
        ) from exc

    data_path = args.snap_patents_path
    if data_path is None:
        data_dir = osp.join(args.root, 'snap-patents')
        os.makedirs(data_dir, exist_ok=True)
        data_path = osp.join(data_dir, 'snap_patents.mat')

    if not osp.exists(data_path):
        try:
            import gdown
        except ImportError as exc:
            raise ImportError(
                'snap-patents download requires gdown. Install it with '
                '`pip install gdown`, or pass --snap-patents-path to a local '
                'snap_patents.mat file.'
            ) from exc

        gdown.download(
            id=SNAP_PATENTS_GDRIVE_ID,
            output=data_path,
            quiet=False
        )

    full_data = scipy.io.loadmat(data_path)
    edge_index = torch.as_tensor(full_data['edge_index'], dtype=torch.long)

    node_feat = full_data['node_feat']
    if hasattr(node_feat, 'todense'):
        node_feat = np.asarray(node_feat.todense())
    x = torch.as_tensor(node_feat, dtype=torch.float)

    num_nodes = int(np.asarray(full_data['num_nodes']).reshape(-1)[0])
    years = np.asarray(full_data['years']).reshape(-1)
    y = torch.as_tensor(
        even_quantile_labels(years, args.year_classes),
        dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    return make_large_dataset('snap-patents', data, args.year_classes, args, transform)


def load_dataset(data_id, args):
    # Keep the original feature normalization for datasets 0~8.  The three
    # large heterophily benchmarks use the raw features from their benchmark
    # loaders, so we do not row-normalize them here.
    transform = None if data_id in [9, 10, 11] else NormalizeFeatures()

    if data_id == 0:
        dataset = Planetoid(
            root=args.root + '/Cora',
            name='Cora',
            split=args.split,
            num_train_per_class=args.train_per_class,
            num_val=args.num_val,
            num_test=args.num_test,
            transform=transform
        )
    elif data_id == 1:
        dataset = Planetoid(
            root=args.root + '/Citeseer',
            name='CiteSeer',
            split=args.split,
            num_train_per_class=args.train_per_class,
            num_val=args.num_val,
            num_test=args.num_test,
            transform=transform
        )
    elif data_id == 2:
        dataset = Planetoid(
            root=args.root + '/Pubmed',
            name='PubMed',
            split=args.split,
            num_train_per_class=args.train_per_class,
            num_val=args.num_val,
            num_test=args.num_test,
            transform=transform
        )
    elif data_id == 3:
        dataset = WikipediaNetwork(
            root=args.root + '/Chameleon',
            name='chameleon',
            transform=transform
        )
    elif data_id == 4:
        dataset = WikipediaNetwork(
            root=args.root + '/Squirrel',
            name='squirrel',
            transform=transform
        )
    elif data_id == 5:
        dataset = Actor(
            root=args.root + '/Actor',
            transform=transform
        )
    elif data_id == 6:
        dataset = WebKB(
            root=args.root + '/Cornell',
            name='Cornell',
            transform=transform
        )
    elif data_id == 7:
        dataset = WebKB(
            root=args.root + '/Texas',
            name='Texas',
            transform=transform
        )
    elif data_id == 8:
        dataset = WebKB(
            root=args.root + '/Wisconsin',
            name='Wisconsin',
            transform=transform
        )
    elif data_id == 9:
        dataset = LINKXDataset(
            root=osp.join(args.root, 'LINKX'),
            name='penn94',
            transform=transform
        )
    elif data_id == 10:
        dataset = load_arxiv_year_dataset(args, transform)
    elif data_id == 11:
        dataset = load_snap_patents_dataset(args, transform)
    else:
        raise ValueError('data must be an integer from 0 to 11')

    data = dataset[0]
    class_count = int(dataset.num_classes)

    # Penn94 ships fixed split masks through LINKXDataset.  For arXiv-year and
    # snap-patents, prefer the public LINKX/GloGNN split files when available;
    # otherwise the random split created in make_large_dataset remains in use.
    maybe_apply_fixed_large_split(data, data_id, args)

    if hasattr(data, 'train_mask') and data.train_mask is not None:
        data.train_mask = select_mask(data.train_mask, args.split_id)
        data.val_mask = select_mask(data.val_mask, args.split_id)
        data.test_mask = select_mask(data.test_mask, args.split_id)
    else:
        if is_large_hetero_dataset(data_id):
            data = random_prop_split(
                data,
                train_prop=args.train_prop,
                val_prop=args.val_prop,
                seed=args.seed,
                ignore_negative=True
            )
        else:
            data = random_balanced_split(
                data,
                class_count,
                args.train_per_class,
                args.num_val,
                args.num_test,
                args.seed
            )

    data = sanitize_labels_and_masks(data, class_count)

    edge_index, _ = remove_self_loops(data.edge_index)

    if args.force_undirected:
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    data.edge_index = edge_index

    return dataset, data, class_count


def sparse_norm_adj(edge_index, num_nodes, device, dropedge=0.0, training=False, add_self_loop=True, norm='sym'):
    edge_index = edge_index.to(device)

    if training and dropedge > 0.0:
        keep = torch.rand(edge_index.size(1), device=device) >= dropedge
        edge_index = edge_index[:, keep]

    if add_self_loop:
        loop = torch.arange(num_nodes, device=device)
        src = torch.cat([edge_index[0], loop], dim=0)
        dst = torch.cat([edge_index[1], loop], dim=0)
    else:
        src = edge_index[0]
        dst = edge_index[1]

    if norm == 'none':
        values = torch.ones(dst.size(0), device=device)
    elif norm == 'row':
        # Row-normalized incoming aggregation: each destination averages its
        # incoming neighbors.  This is used only by the directed arXiv-year and
        # snap-patents presets unless explicitly overridden.
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=device))
        values = deg.clamp_min(1.0).reciprocal()[dst]
    elif norm == 'sym':
        deg = torch.zeros(num_nodes, device=device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=device))

        deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
        values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    else:
        raise ValueError(f'unknown adjacency normalization: {norm}')

    indices = torch.stack([dst, src], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        device=device
    ).coalesce()


def sparse_identity(num_nodes, device):
    idx = torch.arange(num_nodes, device=device)
    return torch.sparse_coo_tensor(
        torch.stack([idx, idx], dim=0),
        torch.ones(num_nodes, device=device),
        (num_nodes, num_nodes),
        device=device
    ).coalesce()


def sparse_identity_minus(adj):
    adj = adj.coalesce()
    num_nodes = adj.size(0)
    device = adj.device
    loop = torch.arange(num_nodes, device=device)
    identity_indices = torch.stack([loop, loop], dim=0)
    indices = torch.cat([identity_indices, adj.indices()], dim=1)
    values = torch.cat([
        torch.ones(num_nodes, device=device),
        -adj.values()
    ], dim=0)
    return torch.sparse_coo_tensor(
        indices,
        values,
        adj.size(),
        device=device
    ).coalesce()


def build_training_adj(args, edge_index, num_nodes, device, dropedge=0.0, training=False, add_self_loop=True):
    if args.model == 'acmgcn':
        low = sparse_norm_adj(
            edge_index,
            num_nodes,
            device,
            dropedge=dropedge,
            training=training,
            add_self_loop=True,
            norm='row'
        )
        high = sparse_identity_minus(low)
        return low, high

    return sparse_norm_adj(
        edge_index,
        num_nodes,
        device,
        dropedge=dropedge,
        training=training,
        add_self_loop=add_self_loop,
        norm=args.adj_norm
    )


def smooth_cross_entropy(logits, target, smoothing):
    if smoothing <= 0.0:
        return F.cross_entropy(logits, target)

    log_prob = F.log_softmax(logits, dim=-1)
    n_class = logits.size(-1)

    with torch.no_grad():
        y = torch.zeros_like(log_prob)
        y.fill_(smoothing / max(1, n_class - 1))
        y.scatter_(1, target.view(-1, 1), 1.0 - smoothing)

    return -(y * log_prob).sum(dim=-1).mean()


def accuracy(logits, y, mask):
    if int(mask.sum()) == 0:
        return 0.0

    pred = logits.argmax(dim=-1)
    return float(pred[mask].eq(y[mask]).sum().item()) / int(mask.sum())


def evaluate(model, data, adj):
    model.eval()

    with torch.no_grad():
        logits = model(data.x, adj)

        train_loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask]
        ).item()
        val_loss = F.cross_entropy(
            logits[data.val_mask],
            data.y[data.val_mask]
        ).item()
        test_loss = F.cross_entropy(
            logits[data.test_mask],
            data.y[data.test_mask]
        ).item()

        train_acc = accuracy(logits, data.y, data.train_mask)
        val_acc = accuracy(logits, data.y, data.val_mask)
        test_acc = accuracy(logits, data.y, data.test_mask)

    return logits, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc


# -----------------------------
# Config
# -----------------------------

def fill_defaults(args):
    """
    Keep homophilic datasets 0~2 on the original GCNII branch.
    Use small-graph heterophily defaults for datasets 3~8 and more
    memory-friendly defaults for large heterophily datasets 9~11.
    """
    if args.model == 'auto':
        if args.data <= 2:
            args.model = 'gcnii'
        elif is_large_hetero_dataset(args.data):
            # LINKX is the strongest simple large-scale default on these
            # benchmarks; MLP/GPR remain available as ablations.
            args.model = 'linkx'
        else:
            args.model = 'h2gcn'

    preset = {}

    if args.model == 'gcnii':
        # Original homophily-oriented GCNII preset.
        preset = {
            'epochs': 1500,
            'patience': 100,
            'hidden': 64,
            'layers': 64,
            'dropout': 0.6,
            'lr': 0.01,
            'wd1': 0.01,
            'wd2': 5e-4,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_loss',
            'propagation_steps': 10,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

        if args.data == 1:
            preset.update({
                'hidden': 256,
                'layers': 32,
                'dropout': 0.7,
                'lamda': 0.6
            })

        if args.data == 2:
            preset.update({
                'hidden': 256,
                'layers': 16,
                'dropout': 0.5,
                'lamda': 0.4,
                'wd1': 5e-4
            })

    elif args.model == 'appnp':
        preset = {
            'epochs': 1500,
            'patience': 200,
            'hidden': 64,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 10,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'dagnn':
        preset = {
            'epochs': 1500,
            'patience': 200,
            'hidden': 128,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 10,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'gcn':
        preset = {
            'epochs': 1000 if is_large_hetero_dataset(args.data) else 1500,
            'patience': 200,
            'hidden': 128,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 0,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'mixhop':
        preset = {
            'epochs': 1000 if is_large_hetero_dataset(args.data) else 1500,
            'patience': 200,
            'hidden': 128 if args.data != 11 else 64,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 2,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'acmgcn':
        preset = {
            'epochs': 500 if is_large_hetero_dataset(args.data) else 1500,
            'patience': 150,
            'hidden': 64 if is_large_hetero_dataset(args.data) else 128,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 1e-3,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 0,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }


    elif args.model == 'gpr':
        hetero = is_hetero_dataset(args.data)
        large = is_large_hetero_dataset(args.data)
        preset = {
            'epochs': 1500 if not large else 1000,
            'patience': 300 if hetero and not large else 200,
            'hidden': 64 if large else (128 if hetero else 64),
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.5 if hetero else 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 5 if large else 10,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'alt' if hetero else 'ppr'
        }

        if args.data in [10, 11]:
            # Large heterophily benchmark setting: directed graph, K=10,
            # alpha=0.1, random GPR coefficients and a BN MLP head.
            preset.update({
                'hidden': 128 if args.data == 10 else 64,
                'alpha': 0.1,
                'propagation_steps': 10,
                'gpr_init': 'random',
                'wd1': 1e-3,
                'label_smoothing': 0.0
            })

    elif args.model == 'mlp':
        preset = {
            'epochs': 1500,
            'patience': 300 if is_hetero_dataset(args.data) else 200,
            'hidden': 256,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 0,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'h2gcn':
        preset = {
            'epochs': 1500,
            'patience': 300,
            'hidden': 128,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 2,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

    elif args.model == 'linkx':
        preset = {
            'epochs': 1500,
            'patience': 300,
            'hidden': 256,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 1e-4,
            'wd2': 0.0,
            'alpha': 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 0,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'ppr'
        }

        if args.data in [9, 10, 11]:
            preset.update({
                'epochs': 1000,
                'patience': 200,
                'hidden': 256 if args.data in [9, 10] else 64,
                'wd1': 1e-3
            })

    else:
        raise ValueError(f'unknown model: {args.model}')

    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


def build_model(args, in_dim, out_dim, num_nodes=None):
    if args.model == 'gcnii':
        return GCNII(
            in_dim,
            args.hidden,
            out_dim,
            args.layers,
            args.dropout,
            args.lamda,
            args.alpha,
            args.variant
        )

    if args.model == 'appnp':
        return APPNPNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.propagation_steps,
            args.alpha,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'dagnn':
        return DAGNNNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.propagation_steps,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'gcn':
        return GCNNet(
            in_dim,
            args.hidden,
            out_dim,
            args.layers,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'mixhop':
        return MixHopNet(
            in_dim,
            args.hidden,
            out_dim,
            args.layers,
            args.propagation_steps,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'acmgcn':
        return ACMGCNNet(
            in_dim,
            args.hidden,
            out_dim,
            args.layers,
            args.dropout,
            use_bn=args.use_bn,
            variant=args.variant,
            plus_plus=args.acm_plus_plus
        )


    if args.model == 'gpr':
        return GPRNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.propagation_steps,
            args.alpha,
            args.dropout,
            args.gpr_init,
            use_bn=args.use_bn
        )

    if args.model == 'mlp':
        return MLPNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'h2gcn':
        return H2GCNLite(
            in_dim,
            args.hidden,
            out_dim,
            args.propagation_steps,
            args.dropout,
            use_bn=args.use_bn
        )

    if args.model == 'linkx':
        if num_nodes is None:
            raise ValueError('num_nodes is required for linkx')

        return LINKXNet(
            in_dim,
            num_nodes,
            args.hidden,
            out_dim,
            args.dropout,
            use_bn=args.use_bn,
            skip=args.linkx_skip,
            init_layers_x=args.linkx_init_layers,
            final_layers=args.mlp_layers,
            inner_activation=args.linkx_inner_activation,
            inner_dropout=args.linkx_inner_dropout
        )

    raise ValueError(f'unknown model: {args.model}')


def build_optimizer(model, args):
    if args.model == 'gcnii':
        return torch.optim.Adam(
            [
                {'params': model.params1, 'weight_decay': args.wd1},
                {'params': model.params2, 'weight_decay': args.wd2}
            ],
            lr=args.lr
        )

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith('bias'):
            no_decay.append(param)
        else:
            decay.append(param)

    return torch.optim.Adam(
        [
            {'params': decay, 'weight_decay': args.wd1},
            {'params': no_decay, 'weight_decay': 0.0}
        ],
        lr=args.lr
    )


def is_better(args, val_loss, val_acc, best_val_loss, best_val_acc):
    if args.selection == 'val_loss':
        return val_loss < best_val_loss

    if args.selection == 'val_acc':
        return val_acc > best_val_acc

    return val_acc > best_val_acc or (
        val_acc == best_val_acc and val_loss < best_val_loss
    )


# -----------------------------
# Training
# -----------------------------

def run_once(run_seed, args, dataset, data, class_count, device):
    set_seed(run_seed)

    local_data = copy.copy(data)
    local_data.x = data.x.to(device)
    local_data.y = data.y.to(device)
    local_data.edge_index = data.edge_index.to(device)
    local_data.train_mask = data.train_mask.to(device)
    local_data.val_mask = data.val_mask.to(device)
    local_data.test_mask = data.test_mask.to(device)

    # Heterophily-specific models should not mix self information into neighbor
    # channels through self-loops. MLP ignores adj anyway.
    add_self_loop = args.model not in ['h2gcn', 'linkx', 'mixhop']

    eval_adj = build_training_adj(
        args,
        local_data.edge_index,
        local_data.num_nodes,
        device,
        dropedge=0.0,
        training=False,
        add_self_loop=add_self_loop
    )

    model = build_model(
        args,
        dataset.num_node_features,
        class_count,
        num_nodes=local_data.num_nodes
    ).to(device)

    optimizer = build_optimizer(model, args)

    best_state = None
    best_epoch = 0
    best_val_loss = float('inf')
    best_val_acc = -1.0
    best_test_acc = 0.0
    bad_epochs = 0

    progress = tqdm(range(args.epochs), disable=args.quiet)
    start_time = time.time()

    for epoch in progress:
        model.train()

        if args.dropedge > 0.0:
            train_adj = build_training_adj(
                args,
                local_data.edge_index,
                local_data.num_nodes,
                device,
                dropedge=args.dropedge,
                training=True,
                add_self_loop=add_self_loop
            )
        else:
            train_adj = eval_adj

        optimizer.zero_grad()

        logits = model(local_data.x, train_adj)
        loss = smooth_cross_entropy(
            logits[local_data.train_mask],
            local_data.y[local_data.train_mask],
            args.label_smoothing
        )

        loss.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            _, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = evaluate(
                model,
                local_data,
                eval_adj
            )

            if is_better(args, val_loss, val_acc, best_val_loss, best_val_acc):
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_test_acc = test_acc
                bad_epochs = 0
            else:
                bad_epochs += args.eval_every

            progress.set_postfix(best_test=best_test_acc)

            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc = evaluate(
        model,
        local_data,
        eval_adj
    )

    elapsed = time.time() - start_time

    return {
        'seed': run_seed,
        'epoch': best_epoch,
        'best_train_loss': train_loss,
        'best_val_loss': val_loss,
        'best_test_loss': test_loss,
        'best_train_acc': train_acc,
        'best_val_acc': val_acc,
        'best_test_acc': test_acc,
        'time': elapsed
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='homophily-preserving baseline with heterophily branch'
    )

    parser.add_argument('data', type=int)

    parser.add_argument(
        '--model',
        type=str,
        default='auto',
        choices=['auto', 'gcnii', 'appnp', 'dagnn', 'gcn', 'mixhop', 'acmgcn', 'gpr', 'mlp', 'h2gcn', 'linkx']
    )

    parser.add_argument('--root', type=str, default='/tmp')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd1', type=float, default=None)
    parser.add_argument('--wd2', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--dropedge', type=float, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runs', type=int, default=1)

    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--split-id', type=int, default=0)
    parser.add_argument('--train-per-class', type=int, default=20)
    parser.add_argument('--num-val', type=int, default=500)
    parser.add_argument('--num-test', type=int, default=1000)
    parser.add_argument('--train-prop', type=float, default=0.5)
    parser.add_argument('--val-prop', type=float, default=0.25)
    parser.add_argument('--year-classes', type=int, default=5)
    parser.add_argument('--snap-patents-path', type=str, default=None)

    parser.add_argument('--force-undirected', dest='force_undirected', action='store_true', default=None)
    parser.add_argument('--keep-directed', dest='force_undirected', action='store_false')
    parser.add_argument(
        '--adj-norm',
        type=str,
        default=None,
        choices=['sym', 'row', 'none'],
        help='Adjacency normalization. Defaults are dataset-specific.'
    )

    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--lamda', type=float, default=None)
    parser.add_argument('--variant', action='store_true', default=False)

    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument(
        '--selection',
        type=str,
        default=None,
        choices=['val_loss', 'val_acc', 'hybrid']
    )

    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--grad-clip', type=float, default=0.0)
    parser.add_argument('--propagation-steps', type=int, default=None)
    parser.add_argument('--mlp-layers', type=int, default=None)
    parser.add_argument('--use-bn', dest='use_bn', action='store_true', default=None)
    parser.add_argument('--no-bn', dest='use_bn', action='store_false')
    parser.add_argument('--linkx-skip', dest='linkx_skip', action='store_true', default=None)
    parser.add_argument('--no-linkx-skip', dest='linkx_skip', action='store_false')
    parser.add_argument('--linkx-init-layers', type=int, default=None)
    parser.add_argument('--linkx-inner-activation', action='store_true', default=False)
    parser.add_argument('--linkx-inner-dropout', action='store_true', default=False)
    parser.add_argument('--acm-plus-plus', dest='acm_plus_plus', action='store_true', default=None)
    parser.add_argument('--no-acm-plus-plus', dest='acm_plus_plus', action='store_false')
    parser.add_argument('--use-fixed-large-splits', dest='use_fixed_large_splits', action='store_true', default=None)
    parser.add_argument('--random-large-splits', dest='use_fixed_large_splits', action='store_false')

    parser.add_argument(
        '--gpr-init',
        type=str,
        default=None,
        choices=['ppr', 'nppr', 'sgc', 'alt', 'random']
    )

    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    args = fill_defaults(args)

    # Dataset-specific switches for the three large heterophily benchmarks only.
    # For data 0~8, these evaluate to the previous defaults: undirected + sym
    # normalized adjacency, no BatchNorm, and no LINKX skip path.
    if args.force_undirected is None:
        args.force_undirected = args.data not in [10, 11]

    if args.adj_norm is None:
        if args.model == 'linkx' and args.data in [9, 10, 11]:
            args.adj_norm = 'none'
        elif args.model == 'acmgcn':
            args.adj_norm = 'row'
        elif args.data in [10, 11]:
            args.adj_norm = 'row'
        else:
            args.adj_norm = 'sym'

    if args.use_bn is None:
        args.use_bn = args.data in [9, 10, 11]

    if args.linkx_skip is None:
        args.linkx_skip = args.model == 'linkx' and args.data in [9, 10, 11]

    if args.linkx_init_layers is None:
        args.linkx_init_layers = 1

    if args.acm_plus_plus is None:
        args.acm_plus_plus = args.model == 'acmgcn'

    if args.use_fixed_large_splits is None:
        args.use_fixed_large_splits = args.data in [10, 11]

    if args.weight_decay is not None:
        args.wd1 = args.weight_decay
        args.wd2 = args.weight_decay

    if not (0.0 < args.train_prop < 1.0):
        raise ValueError('--train-prop must be in (0, 1)')
    if not (0.0 <= args.val_prop < 1.0):
        raise ValueError('--val-prop must be in [0, 1)')
    if args.train_prop + args.val_prop >= 1.0:
        raise ValueError('--train-prop + --val-prop must be < 1')
    if args.linkx_init_layers < 1:
        raise ValueError('--linkx-init-layers must be >= 1')

    return args


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    dataset, data, class_count = load_dataset(args.data, args)

    results = []

    for run in range(args.runs):
        result = run_once(
            args.seed + run,
            args,
            dataset,
            data,
            class_count,
            device
        )
        results.append(result)

        print(
            'run', run,
            'seed', result['seed'],
            'best_epoch', result['epoch'],
            'best_train_acc', result['best_train_acc'],
            'best_val_acc', result['best_val_acc'],
            'best_test_acc', result['best_test_acc'],
            'time', result['time']
        )

    vals = np.array([r['best_val_acc'] for r in results], dtype=np.float64)
    tests = np.array([r['best_test_acc'] for r in results], dtype=np.float64)
    times = np.array([r['time'] for r in results], dtype=np.float64)

    print('dataset', getattr(dataset, 'name', dataset.__class__.__name__))
    print('model', args.model)
    print('split', args.split)
    print('selection', args.selection)
    print('mean_best_val_acc', float(vals.mean()))
    print('std_best_val_acc', float(vals.std()))
    print('mean_val_selected_test_acc', float(tests.mean()))
    print('std_val_selected_test_acc', float(tests.std()))
    print('mean_time', float(times.mean()))
    print('config', vars(args))


if __name__ == '__main__':
    main()
