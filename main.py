import argparse
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.datasets import Actor, Planetoid, WebKB, WikipediaNetwork
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
    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout):
        super().__init__()
        modules = []

        if layers <= 1:
            modules.append(nn.Linear(in_dim, out_dim))
        else:
            modules.append(nn.Linear(in_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))

            for _ in range(layers - 2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
                modules.append(nn.Dropout(dropout))

            modules.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MLPNet(nn.Module):
    """MLP baseline with the same forward signature as graph models."""

    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, layers, dropout)

    def forward(self, x, adj=None):
        return self.mlp(x)


class APPNPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, alpha, dropout):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout)
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
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, dropout):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout)
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
    def __init__(self, in_dim, hidden_dim, out_dim, mlp_layers, propagation_steps, alpha, dropout, init):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, out_dim, mlp_layers, dropout)
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

    def __init__(self, in_dim, hidden_dim, out_dim, hops, dropout):
        super().__init__()
        self.hops = hops
        self.dropout = dropout

        self.lin = nn.Linear(in_dim, hidden_dim)
        self.pred = MLP(hidden_dim * (hops + 1), hidden_dim, out_dim, 2, dropout)

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
    Lightweight LINKX-style model.

    It separately embeds adjacency structure and node features, then combines them.
    This is useful for non-homophilous graphs where feature and topology signals
    should not be forced through the same smoothing operator.
    """

    def __init__(self, in_dim, num_nodes, hidden_dim, out_dim, dropout):
        super().__init__()
        self.dropout = dropout

        self.adj_weight = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        self.adj_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.x_mlp = MLP(in_dim, hidden_dim, hidden_dim, 2, dropout)
        self.final_mlp = MLP(hidden_dim * 2, hidden_dim, out_dim, 2, dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.adj_weight)
        nn.init.zeros_(self.adj_bias)

    def forward(self, x, adj):
        a_emb = torch.sparse.mm(adj, self.adj_weight) + self.adj_bias
        a_emb = F.relu(a_emb)

        x_emb = self.x_mlp(F.dropout(x, self.dropout, training=self.training))

        h = torch.cat([a_emb, x_emb], dim=1)
        h = F.dropout(h, self.dropout, training=self.training)
        return self.final_mlp(h)


# -----------------------------
# Utilities
# -----------------------------

def is_hetero_dataset(data_id):
    return data_id >= 3


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


def load_dataset(data_id, args):
    transform = NormalizeFeatures()

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
    else:
        raise ValueError('data must be an integer from 0 to 8')

    data = dataset[0]
    class_count = int(dataset.num_classes)

    # Defensive handling for rare invalid labels.
    data.y = torch.where(data.y > class_count - 1, torch.zeros_like(data.y), data.y)

    if hasattr(data, 'train_mask') and data.train_mask is not None:
        data.train_mask = select_mask(data.train_mask, args.split_id)
        data.val_mask = select_mask(data.val_mask, args.split_id)
        data.test_mask = select_mask(data.test_mask, args.split_id)
    else:
        data = random_balanced_split(
            data,
            class_count,
            args.train_per_class,
            args.num_val,
            args.num_test,
            args.seed
        )

    edge_index, _ = remove_self_loops(data.edge_index)

    if args.force_undirected:
        edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

    data.edge_index = edge_index

    return dataset, data, class_count


def sparse_norm_adj(edge_index, num_nodes, device, dropedge=0.0, training=False, add_self_loop=True):
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

    deg = torch.zeros(num_nodes, device=device)
    deg.index_add_(0, dst, torch.ones(dst.size(0), device=device))

    deg_inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
    values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([dst, src], dim=0)

    return torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes),
        device=device
    ).coalesce()


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
    Use a separate heterophilic branch for datasets 3~8.
    """
    if args.model == 'auto':
        args.model = 'gcnii' if args.data <= 2 else 'h2gcn'

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

    elif args.model == 'gpr':
        hetero = is_hetero_dataset(args.data)
        preset = {
            'epochs': 1500,
            'patience': 300 if hetero else 200,
            'hidden': 128 if hetero else 64,
            'layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'wd1': 5e-4,
            'wd2': 0.0,
            'alpha': 0.5 if hetero else 0.1,
            'lamda': 0.5,
            'selection': 'val_acc',
            'propagation_steps': 10,
            'mlp_layers': 2,
            'label_smoothing': 0.0,
            'dropedge': 0.0,
            'gpr_init': 'alt' if hetero else 'ppr'
        }

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
            args.dropout
        )

    if args.model == 'dagnn':
        return DAGNNNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.propagation_steps,
            args.dropout
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
            args.gpr_init
        )

    if args.model == 'mlp':
        return MLPNet(
            in_dim,
            args.hidden,
            out_dim,
            args.mlp_layers,
            args.dropout
        )

    if args.model == 'h2gcn':
        return H2GCNLite(
            in_dim,
            args.hidden,
            out_dim,
            args.propagation_steps,
            args.dropout
        )

    if args.model == 'linkx':
        if num_nodes is None:
            raise ValueError('num_nodes is required for linkx')

        return LINKXNet(
            in_dim,
            num_nodes,
            args.hidden,
            out_dim,
            args.dropout
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
    add_self_loop = args.model not in ['h2gcn', 'linkx']

    eval_adj = sparse_norm_adj(
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
            train_adj = sparse_norm_adj(
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
        choices=['auto', 'gcnii', 'appnp', 'dagnn', 'gpr', 'mlp', 'h2gcn', 'linkx']
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

    parser.add_argument('--force-undirected', action='store_true', default=True)
    parser.add_argument('--keep-directed', dest='force_undirected', action='store_false')

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

    parser.add_argument(
        '--gpr-init',
        type=str,
        default=None,
        choices=['ppr', 'nppr', 'sgc', 'alt']
    )

    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()
    args = fill_defaults(args)

    if args.weight_decay is not None:
        args.wd1 = args.weight_decay
        args.wd2 = args.weight_decay

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

    print('dataset', dataset.__class__.__name__)
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
