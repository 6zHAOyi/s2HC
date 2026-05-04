from __future__ import annotations
from typing import Callable

from functools import partial
from random import randrange
import math
import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Sequential
from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange, Reduce

"""
ein notation:
b - batch
d - feature dimension
s - residual streams
t - residual streams + num branch inputs
f - number of fractions (division of feature dimension space)
v - number of views for branch input
"""



# helper functions
def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

class SHyperConnections(nn.Module):
    def __init__(
        self,
        num_residual_streams,
        *,
        dim,
        branch: nn.Module | None = None,
        layer_index = None,
        channel_first = False,
        dropout = 0.,
        residual_transform: nn.Module | None = None,
        add_branch_out_to_residual = True,
        depth_residual_fn = add,
    ):
        super().__init__()

        self.branch = branch
        self.num_residual_streams = num_residual_streams
        self.channel_first = channel_first
        s = num_residual_streams 
        sm1 = s - 1
        
        init_residual_index = default(layer_index, randrange(s)) % s
        in_dim = int(dim * s)
        self.norm = RMSNorm(in_dim)

        J = torch.ones(s, s) / s
        self.register_buffer('J', J)
        
        Uz = torch.zeros(s, sm1)
        for i in range(1, s):
            val = 1.0 / math.sqrt(i * (i + 1))
            Uz[:i, i - 1] = val
            Uz[i, i - 1] = -i * val
        self.register_buffer('Uz', Uz)
        

        I_sm1 = torch.eye(sm1)
        self.register_buffer('eye_sm1', I_sm1) 
        
        triu_i, triu_j = torch.triu_indices(sm1, sm1, offset=1)
        self.register_buffer('triu_i', triu_i)
        self.register_buffer('triu_j', triu_j)

        # ---------------------------------------------------------------------
        # Pre
        # ---------------------------------------------------------------------
        self.to_alpha_pre = nn.Linear(in_dim, s, bias=False)
        nn.init.zeros_(self.to_alpha_pre.weight)
        init_alpha0 = torch.ones((s, 1)) * -1
        init_alpha0[init_residual_index, 0] = 1.
        self.static_alpha_pre = nn.Parameter(init_alpha0)

        # ---------------------------------------------------------------------
        # Residual
        # ---------------------------------------------------------------------
        self.to_alpha_residual = nn.Linear(in_dim, sm1**2, bias=False)
        nn.init.zeros_(self.to_alpha_residual.weight)
        init_bias = torch.zeros(sm1**2)

        init_bias[-sm1:] = 4.0 
        self.static_alpha_residual = nn.Parameter(init_bias)
        
        self.pre_branch_scale = nn.Parameter(torch.ones(1) * 1e-2)
        self.residual_rot_scale_u = nn.Parameter(torch.ones(1) * 1e-2) 
        self.residual_rot_scale_v = nn.Parameter(torch.ones(1) * 1e-2) 
        self.residual_val_scale = nn.Parameter(torch.ones(1) * 1e-2) 

        # U, V
        self.gamma_u = nn.Parameter(torch.ones(1))
        self.gamma_v = nn.Parameter(torch.ones(1))

        # ---------------------------------------------------------------------
        # Beta
        # ---------------------------------------------------------------------
        self.add_branch_out_to_residual = add_branch_out_to_residual
        if add_branch_out_to_residual:
            self.to_beta = nn.Linear(in_dim, s, bias=False)
            nn.init.zeros_(self.to_beta.weight)
            beta_init = torch.ones(s) * -1.
            beta_init[init_residual_index] = 1.
            self.static_beta = nn.Parameter(beta_init)
            self.h_post_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.dropout = nn.Dropout(dropout)
        self.residual_transform = default(residual_transform, nn.Identity())
        self.depth_residual_fn = depth_residual_fn

    def width_connection(self, residuals):
        s = self.num_residual_streams
        sm1 = s - 1

        maybe_transformed_residuals = self.residual_transform(residuals)
        if self.channel_first:
            residuals = rearrange(residuals, 'b d ... -> b ... d')
        residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s=s)
        normed = rearrange(residuals, 'b ... s d -> b ... (s d)', s=s)
        normed = self.norm(normed)

        # Pre
        dynamic_pre = self.to_alpha_pre(normed)
        dynamic_pre = rearrange(dynamic_pre, '... s -> ... s 1')
        alpha_pre = (dynamic_pre * self.pre_branch_scale) + self.static_alpha_pre
        alpha_pre = alpha_pre.sigmoid()

        # Residual
        dynamic_res = self.to_alpha_residual(normed) 
        
        k = sm1 * (sm1 - 1) // 2
        dyn_U = dynamic_res[..., :k] * self.residual_rot_scale_u
        dyn_V = dynamic_res[..., k:2*k] * self.residual_rot_scale_v
        dyn_S = dynamic_res[..., 2*k:] * self.residual_val_scale

        stat_U = self.static_alpha_residual[:k]
        stat_V = self.static_alpha_residual[k:2*k]
        stat_S = self.static_alpha_residual[2*k:]

        z_U = self.gamma_u * torch.tanh(dyn_U + stat_U)
        z_V = self.gamma_v * torch.tanh(dyn_V + stat_V)
        sigma = torch.tanh(dyn_S + stat_S)
        
        A_U = torch.zeros(*z_U.shape[:-1], sm1, sm1, device=normed.device, dtype=normed.dtype)
        A_V = torch.zeros(*z_V.shape[:-1], sm1, sm1, device=normed.device, dtype=normed.dtype)
        
        A_U[..., self.triu_i, self.triu_j] = z_U
        A_V[..., self.triu_i, self.triu_j] = z_V
        
        A_U = A_U - A_U.transpose(-1, -2)
        A_V = A_V - A_V.transpose(-1, -2)
        
        U_core = torch.linalg.solve(self.eye_sm1 + A_U, self.eye_sm1 - A_U)
        V_core = torch.linalg.solve(self.eye_sm1 + A_V, self.eye_sm1 - A_V)
        
        E_core = U_core @ (sigma.unsqueeze(-1) * V_core.transpose(-1, -2))
        X = self.Uz @ E_core @ self.Uz.T
        H = self.J + X
        alpha_residual = H
        
        alpha = cat((alpha_pre, alpha_residual), dim=-1)

        beta = None
        if self.add_branch_out_to_residual:
            dc_weight = self.to_beta(normed)
            beta = (dc_weight * self.h_post_scale) + self.static_beta
            beta = beta.sigmoid() * 2 

        mix_h = einsum(alpha, residuals, '... s t, ... s d -> ... t d')
        branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]

        if self.channel_first:
            branch_input = rearrange(branch_input, 'b ... d -> b d ...')
        
        residuals = rearrange(residuals, 'b ... s d -> (b s) ... d')
        
        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return branch_input, residuals, dict(beta=beta)

    def depth_connection(self, branch_output, residuals, *, beta):
        assert self.add_branch_out_to_residual

        if self.channel_first:
            branch_output = rearrange(branch_output, 'b d t -> b t d')

        output = einsum(branch_output, beta, 'b t d, b t s -> b t s d')
        output = rearrange(output, 'b t s d -> (b s) t d')

        if self.channel_first:
            output = rearrange(output, '(b s) t d -> (b s) d t', s=self.num_residual_streams)

        residuals = self.depth_residual_fn(output, residuals)
        return self.dropout(residuals)

    def decorate_branch(self, branch: Callable):
        assert not exists(self.branch), 'branch was already wrapped on init'
        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)
            branch_output = branch(branch_input, *args, **kwargs)
            residual = add_residual(branch_output)
            return residual
        return forward_and_add_residual

    def forward(self, residuals, *branch_args, **branch_kwargs):
        branch_input, residuals, residual_kwargs = self.width_connection(residuals)
        def add_residual_fn(branch_out):
            if not self.add_branch_out_to_residual:
                return branch_out
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)
            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)
            return tree_unflatten((branch_out, *rest), tree_spec)
        if not exists(self.branch):
            return branch_input, add_residual_fn
        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)
        return add_residual_fn(branch_output)


# main functions

def get_expand_reduce_stream_functions(
    num_streams,
    add_stream_embed = False,
    dim = None,
    disable = False
):
    if num_streams == 1 or disable:
        return (nn.Identity(), nn.Identity())

    if add_stream_embed:
        assert exists(dim), '`dim` must be passed into get_init_and_expand_reduce_stream_functions for returning an expansion function with stream embeddings added'

        expand_fn = StreamEmbed(num_streams, dim, expand_to_streams = True)
    else:
        expand_fn = Reduce(pattern = 'b ... -> (b s) ...', reduction = 'repeat', s = num_streams)

    reduce_fn = Reduce(pattern = '(b s) ... -> b ...', reduction = 'sum', s = num_streams)

    return expand_fn, reduce_fn

def get_init_and_expand_reduce_stream_functions(
    num_streams,
    num_fracs = 1,
    dim = None,
    add_stream_embed = False,
    disable = None,
    **kwargs
):
    disable = default(disable, num_streams == 1)

    hyper_conn_klass = SHyperConnections if not disable else Residual

    init_hyper_conn_fn = partial(hyper_conn_klass, num_streams, **kwargs)
    expand_reduce_fns = get_expand_reduce_stream_functions(num_streams, add_stream_embed = add_stream_embed, dim = dim, disable = disable)

    if exists(dim):
        init_hyper_conn_fn = partial(init_hyper_conn_fn, dim = dim)

    return (init_hyper_conn_fn, *expand_reduce_fns)



# main classes

# residual base class

class Residual(Module):
    def __init__(
        self,
        *args,
        branch: Module | None = None,
        residual_transform: Module | None = None,
        **kwargs
    ):
        super().__init__()
        self.branch = branch
        self.residual_transform = default(residual_transform, nn.Identity())

    def width_connection(
        self,
        residuals
    ):
        return residuals, residuals, dict()

    def depth_connection(
        self,
        branch_output,
        residuals,

    ):
        return branch_output + self.residual_transform(residuals)

    def decorate_branch(
        self,
        branch: Callable
    ):
        assert not exists(self.branch), 'branch was already wrapped on init'

        def forward_and_add_residual(residual, *args, **kwargs):
            branch_input, add_residual = self.forward(residual)

            branch_output = branch(branch_input, *args, **kwargs)

            residual = add_residual(branch_output)

            return residual

        return forward_and_add_residual

    def forward(
        self,
        residuals,
        *branch_args,
        **branch_kwargs
    ):

        branch_input, residuals, residual_kwargs = self.width_connection(residuals)

        def add_residual_fn(branch_out):
            (branch_out, *rest), tree_spec = tree_flatten(branch_out)

            branch_out = self.depth_connection(branch_out, residuals, **residual_kwargs)

            return tree_unflatten((branch_out, *rest), tree_spec)

        if not exists(self.branch):
            return branch_input, add_residual_fn

        branch_output = self.branch(branch_input, *branch_args, **branch_kwargs)

        return add_residual_fn(branch_output)


SHyperConnections.get_expand_reduce_stream_functions = staticmethod(get_expand_reduce_stream_functions)
SHyperConnections.get_init_and_expand_reduce_stream_functions = staticmethod(get_init_and_expand_reduce_stream_functions)

# stream embed

class StreamEmbed(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False,
        expand_to_streams = False
    ):
        super().__init__()
        self.channel_first = channel_first
        self.num_streams = num_streams

        self.expand_to_streams = expand_to_streams
        self.stream_embed = nn.Parameter(torch.zeros(num_streams, dim))

    def forward(self, residuals):

        if self.expand_to_streams:
            residuals = repeat(residuals, 'b ... -> (b s) ...', s = self.num_streams)

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        residuals = residuals + self.stream_embed

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... s d -> (b s) d ...', s = self.num_streams)
        else:
            residuals = rearrange(residuals, 'b ... s d -> (b s) ... d', s = self.num_streams)

        return residuals

# attention pool - taken from Enformer https://www.nature.com/articles/s41592-021-01252-x , in turn taken from somewhere else

class AttentionPoolReduceStream(Module):
    def __init__(
        self,
        num_streams,
        dim,
        channel_first = False
    ):
        super().__init__()
        self.num_streams = num_streams
        self.channel_first = channel_first

        self.to_attn_logits = nn.Linear(dim, dim, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim))

    def forward(self, residuals):

        if self.channel_first:
            residuals = rearrange(residuals, '(b s) d ... -> b ... s d', s = self.num_streams)
        else:
            residuals = rearrange(residuals, '(b s) ... d -> b ... s d', s = self.num_streams)

        attn_logits = self.to_attn_logits(residuals)
        attn = attn_logits.softmax(dim = -2)

        residuals = reduce(residuals * attn, 'b ... s d -> b ... d', 'sum')

        if self.channel_first:
            residuals = rearrange(residuals, 'b ... d -> b d ...')

        return residuals