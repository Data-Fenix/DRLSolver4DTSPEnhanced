import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from typing import NamedTuple
import warnings
from heuristics import HeuristicLogits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.bool).view(*mask.size()[:-1], -1)[..., :n]

def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0

def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

def get_costs(dataset, pi, state, mat):
    depots = torch.zeros(pi.size(0), 1).long().to(device)
    _,ind = torch.max(dataset, dim=2)
    #bdd = mat.var[state.prev_a.squeeze() * mat.n_c].unsqueeze(1)
    bdd = mat.var[state.prev_a.squeeze() * mat.n_c].view(-1, 1)
    bdd = torch.randn(state.prev_a.size(0), device=device) * bdd
    #add = mat.__getd__(ind, state.prev_a, depots, state.lengths).unsqueeze(1)
    add = mat.__getd__(ind, state.prev_a, depots, state.lengths).view(-1, 1)
    bdd = bdd.repeat(1, 2)
    bdd[:,1] = add.squeeze() * 5
    bdd = torch.min(bdd, dim=1)[0]
    bdd = bdd[:, None].repeat(1, 2)
    bdd[:,1] = add.squeeze() * -0.9
    bdd = torch.max(bdd, dim=1)[0]
    return state.lengths.squeeze() + add.squeeze() + bdd.squeeze(), None

# Create the base NamedTuple without methods
class _StateTSPBase(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

# Create the full class that inherits from the base
class StateTSP(_StateTSPBase):
    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                first_a=self.first_a[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            )
        return super(StateTSP, self).__getitem__(key)

    @staticmethod
    def initialize(loc, visited_dtype=torch.bool):
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):
        assert self.all_finished()
        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def addmask(self):
        visited_ = self.visited_.scatter(-1, self.first_a[:, :, None], 1)
        return self._replace(visited_=visited_)        

    def update(self, selected, mat, input):
        """
        Updates decoder state after each step, including current tour time.
        """
        # Update the state in the decoding process
        prev_a = selected[:, None]  # Add dimension for step

        _,ind = torch.max(input, dim=2)
        #bdd = mat.var[self.prev_a.squeeze() * mat.n_c + prev_a.squeeze()].unsqueeze(1)
        bdd = mat.var[self.prev_a.squeeze() * mat.n_c + prev_a.squeeze()].view(-1, 1)
        # Calculate the distance between the previous and selected nodes
        #add = mat.__getd__(ind, self.prev_a, prev_a, self.lengths).unsqueeze(1)
        add = mat.__getd__(ind, self.prev_a, prev_a, self.lengths).view(-1, 1)
        bdd = torch.randn(prev_a.size(0), 1, device=device) * bdd
        bdd = bdd.repeat(1, 2)
        bdd[:, 1] = add.squeeze() * 5
        bdd = torch.min(bdd, dim=1)[0]
        bdd = bdd[:, None].repeat(1, 2)
        bdd[:, 1] = add.squeeze() * -0.9
        bdd = torch.max(bdd, dim=1)[0]
        # Update the length of the tour : Normalized tour time ; tracking the current time in the decoder
        lengths = self.lengths + add + bdd[:, None]

        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        return self._replace(prev_a=prev_a, visited_=visited_, lengths=lengths, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2) - 1

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_

# Create the base NamedTuple without methods
class _AttentionModelFixedBase(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

# Create the full class that inherits from the base
class AttentionModelFixed(_AttentionModelFixedBase):
    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 input_size=4,
                 max_t=12,
                 step_mlp_dim=64,  #   parameter -  MLP for step embedding
                 use_step_mlp=True, #   parameter - MLP for step embedding
                 use_temp_mlp=False,  #  : Control temperature MLP separately
                 # Individual feature flags (for granular control)
                 use_step_ratio=False,
                 use_last_3_nodes=False,
                 use_visited_mean=False,
                 use_unvisited_mean=False,
                 use_sin_cos_time=False,
                 use_linear_time=False, #  : Use linear time encoding instead of sin/cos (1 dim vs 2 dim, individual flag)
                 use_depot_distance=False,
                 use_tour_length=False,
                 use_mean_dist_unvisited=False,
                 # Preset flags (for convenience - will be mapped to individual flags)
                 step_features_v1=False,
                 step_features_v2=False,
                 step_features_v1_light=False,
                 step_features_v2_light=False,
                 step_features_minimal=False,  #   step_ratio + visited_mean only
                 use_cost_aware_gating=True,  # NEW - Cost-aware gating for heuristic blending
                 heuristic_type='linear_time',  # NEW - Type of heuristic blending
                 lambda_heuristic=1.0,  # NEW - Weight for heuristic blending
                 use_nonlinear_transform=False,  # NEW - Nonlinear transform for heuristic blending
                 transform_type='piecewise',  # NEW - Type of nonlinear transform
                 # NEW TIME SLICING PARAMETERS  
                use_time_slicing=False,           # Enable time slicing
                window_size_W=12,                 # Window size in bins
                n_cities=100,                      # Number of cities (for dynamic sizing)
                use_decoder_mlp=False,  # Flag to enable/disable decoder MLP
                decoder_mlp_hidden=512,  # Hidden dimension for decoder MLP (match encoder)
                use_decoder_mlp_pre=False,  # Option C: Pre-attention MLP
                decoder_mlp_pre_hidden=512,  # Hidden dimension for pre-attention MLP
                lambda_heuristic_learnable=False,  # Whether lambda is learnable or fixed
                ):
        
        super(AttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.beam_width = 1  # Default to 1 (greedy), will be set when beam search is used
        self.temp = 1.0
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        step_context_dim = 4 * embedding_dim  # Embedding of first and last node
        node_dim = n_cities  # Input is one-hot over city pool (size n_cities)
        #  : Store time slicing parameters
        self.use_time_slicing = use_time_slicing
        self.window_size_W = window_size_W
        self.max_t = max_t
        self.n_cities = n_cities

        # Store time slicing parameters
        self.use_time_slicing = use_time_slicing
        self.window_size_W = window_size_W
        self.max_t = max_t
        self.n_cities = n_cities
        
        # Window state tracking for Safe Refresh
        # These will be set during forward() and checked during decoding
        self.current_window_start_bin = None  # Current window start (time bin index)
        self.current_window_end_bin = None   # Current window end (time bin index)
        self.last_refresh_time = None         # Last time we refreshed (normalized time)
        self.refresh_count = 0                # Track how many times we've refreshed
        self.refresh_strategy = None          # Will be set from opts
        self.refresh_interval = None          # Will be set from opts
        self.buffer_k_moves = None            # Will be set from opts

        self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
        self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        # Create appropriate traffic embedding layer based on time slicing
        if use_time_slicing:
            # Windowed traffic embedding: Use max possible size (12 bins) to handle forward windows
            # Forward window (W=-1) uses all bins from start_time to end, which can be up to 12 bins
            max_window_size = max_t  # Maximum possible window (full day = 12 bins)
            self.embed_windowed_traffic = nn.Linear(node_dim * max_window_size, embedding_dim)
            print(f"[Time Slicing] Created windowed traffic embedding: {node_dim * max_window_size} → {embedding_dim} (supports forward windows)")
            
            # Keep original for comparison/fallback
            self.embed_static_traffic = nn.Linear(node_dim * max_t, embedding_dim)
        else:
            # Original full time series embedding
            self.embed_static_traffic = nn.Linear(node_dim * max_t, embedding_dim)
        
        #self.embed_static_traffic = nn.Linear(node_dim * max_t, embedding_dim)
        self.embed_static = nn.Linear(2 * embedding_dim, embedding_dim)
        assert embedding_dim % n_heads == 0

        # Decoder MLP components (Option 1: Post-attention MLP)
        self.use_decoder_mlp = use_decoder_mlp
        self.decoder_mlp_hidden = decoder_mlp_hidden

        if self.use_decoder_mlp:
            self.decoder_mlp = nn.Sequential(
                nn.Linear(embedding_dim, decoder_mlp_hidden),
                nn.ReLU(),
                nn.Linear(decoder_mlp_hidden, embedding_dim)
            )
            
            def _init_decoder_mlp_weights(m):
                if isinstance(m, nn.Linear):
                    stdv = 1. / math.sqrt(m.weight.size(0))
                    nn.init.uniform_(m.weight, -stdv, stdv)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            
            self.decoder_mlp.apply(_init_decoder_mlp_weights)
            print(f"[Decoder MLP] Enabled with hidden_dim={decoder_mlp_hidden}")
        else:
            self.decoder_mlp = None
            print("[Decoder MLP] Disabled (baseline mode)") 

        # Decoder MLP Pre-Attention components (Option C: Pre-attention MLP)
        self.use_decoder_mlp_pre = use_decoder_mlp_pre
        self.decoder_mlp_pre_hidden = decoder_mlp_pre_hidden

        if self.use_decoder_mlp_pre:
            self.decoder_mlp_pre = nn.Sequential(
                nn.Linear(embedding_dim, decoder_mlp_pre_hidden),
                nn.ReLU(),
                nn.Linear(decoder_mlp_pre_hidden, embedding_dim)
            )
            
            def _init_decoder_mlp_pre_weights(m):
                if isinstance(m, nn.Linear):
                    stdv = 1. / math.sqrt(m.weight.size(0))
                    nn.init.uniform_(m.weight, -stdv, stdv)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            
            self.decoder_mlp_pre.apply(_init_decoder_mlp_pre_weights)
            print(f"[Decoder MLP Pre-Attention] Enabled with hidden_dim={decoder_mlp_pre_hidden}")
        else:
            self.decoder_mlp_pre = None
            print("[Decoder MLP Pre-Attention] Disabled")


        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        #self.project_traffic = nn.Linear(input_size*input_size, embedding_dim, bias=False)
        #self.project_visit = nn.Linear(input_size, embedding_dim, bias=False)
        #self.xx = torch.tensor([[i for j in range(input_size)] for i in range(input_size)], device=device).view(1, input_size, input_size)
        #self.yy = torch.tensor([[j for j in range(input_size)] for i in range(input_size)], device=device).view(1, input_size, input_size)
        # Use adaptive pooling instead of fixed-size Linear layers
        # Size-agnostic projections using adaptive pooling
        # These can handle variable graph_size at runtime
        # project_traffic: (batch, 1, graph_size^2) -> (batch, 1, embedding_dim)
        self.project_traffic = nn.Sequential(
            nn.AdaptiveAvgPool1d(embedding_dim),  # Pool variable size to embedding_dim
            nn.Linear(embedding_dim, embedding_dim, bias=False)
        )

        # project_visit: (batch, 1, graph_size) -> (batch, 1, embedding_dim)
        self.project_visit = nn.Sequential(
            nn.AdaptiveAvgPool1d(embedding_dim),  # Pool variable size to embedding_dim
            nn.Linear(embedding_dim, embedding_dim, bias=False)
        )
        
        #  : Step-MLP components with granular feature control
        self.use_step_mlp = use_step_mlp
        self.use_temp_mlp = use_temp_mlp
        
        # Store individual feature flags
        self.use_step_ratio = use_step_ratio
        self.use_last_3_nodes = use_last_3_nodes
        self.use_visited_mean = use_visited_mean
        self.use_unvisited_mean = use_unvisited_mean
        self.use_sin_cos_time = use_sin_cos_time
        self.use_linear_time = use_linear_time # Use linear time encoding instead of sin/cos (1 dim vs 2 dim, individual flag)
        self.use_depot_distance = use_depot_distance
        self.use_tour_length = use_tour_length
        self.use_mean_dist_unvisited = use_mean_dist_unvisited

        # Safety check: cannot use both sin/cos time and linear time
        if self.use_sin_cos_time and self.use_linear_time:
            raise ValueError("Cannot use both use_sin_cos_time and use_linear_time. Choose one.")
        
        # Map presets to individual flags (if presets are used)
        if step_features_v1:
            self.use_step_ratio = True
            self.use_last_3_nodes = True
            self.use_visited_mean = True
        if step_features_v2:
            self.use_step_ratio = True
            self.use_sin_cos_time = True
            self.use_depot_distance = True
            self.use_tour_length = True
            self.use_unvisited_mean = True
            self.use_mean_dist_unvisited = True
        if step_features_v1_light:
            self.use_step_ratio = True
            self.use_last_3_nodes = True
            # NO visited_mean
        if step_features_v2_light:
            self.use_step_ratio = True
            self.use_sin_cos_time = True
            self.use_depot_distance = True
            self.use_tour_length = True
            self.use_mean_dist_unvisited = True
            # NO unvisited_mean
        if step_features_minimal:  # step_ratio + visited_mean only
            self.use_step_ratio = True
            self.use_visited_mean = True
        
        # Calculate step_input_dim if either step_mlp or temp_mlp is enabled
        if self.use_step_mlp or self.use_temp_mlp:
            step_input_dim = self._compute_step_input_dim()
            
            # Safety check: at least one feature must be enabled
            if step_input_dim == 0:
                raise ValueError("At least one step feature flag must be enabled when using step_mlp or temp_mlp. "
                            "Enable one of: use_step_ratio, use_linear_time, use_sin_cos_time, use_depot_distance, "
                            "use_tour_length, use_visited_mean, use_unvisited_mean, use_last_3_nodes, "
                            "use_mean_dist_unvisited, or use a preset like step_features_v1, step_features_v2, etc.")
            
            # Initialize Step-MLP if enabled
            if self.use_step_mlp:
                self.step_mlp = nn.Sequential(
                    nn.Linear(step_input_dim, step_mlp_dim),
                    nn.ReLU(),
                    nn.Linear(step_mlp_dim, step_mlp_dim),
                    nn.ReLU(),
                    nn.Linear(step_mlp_dim, embedding_dim)  # Context nudge
                )
                # Initialize Step-MLP weights using the same scheme as original code
                def _init_mlp_weights(m):
                    if isinstance(m, nn.Linear):
                        stdv = 1. / math.sqrt(m.weight.size(0))
                        nn.init.uniform_(m.weight, -stdv, stdv)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                self.step_mlp.apply(_init_mlp_weights)
            else:
                self.step_mlp = None
            
            # Temperature MLP can be created independently or together with step_mlp
            if self.use_temp_mlp:
                self.temp_mlp = nn.Sequential(
                    nn.Linear(step_input_dim, step_mlp_dim),
                    nn.ReLU(),
                    nn.Linear(step_mlp_dim, 1),
                    nn.Sigmoid()  # Output between 0-1
                )
                def _init_mlp_weights(m):
                    if isinstance(m, nn.Linear):
                        stdv = 1. / math.sqrt(m.weight.size(0))
                        nn.init.uniform_(m.weight, -stdv, stdv)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0.0)
                self.temp_mlp.apply(_init_mlp_weights)
            else:
                self.temp_mlp = None
        else:
            # Neither step_mlp nor temp_mlp is enabled
            self.step_mlp = None
            self.temp_mlp = None
        
        # Cost-Aware Gating components
        self.use_cost_aware_gating = use_cost_aware_gating
        if self.use_cost_aware_gating:
            self.heuristic_computer = HeuristicLogits(heuristic_type)
            # Make lambda_heuristic learnable or fixed based on flag
            if lambda_heuristic_learnable:
                self.lambda_heuristic = nn.Parameter(torch.tensor(lambda_heuristic))
            else:
                # Store as a buffer (non-learnable, but moves with model to device)
                self.register_buffer('lambda_heuristic', torch.tensor(lambda_heuristic))
            
            if use_nonlinear_transform:
                self.transform_type = transform_type
                if transform_type == 'piecewise':
                    # Piecewise linear transformation parameters
                    self.transform = nn.Sequential(
                        nn.Linear(1, 16),
                        nn.ReLU(),
                        nn.Linear(16, 1)
                    )
                elif transform_type == 'exponential':
                    # Exponential transformation parameters
                    self.exp_scale = nn.Parameter(torch.tensor(1.0))
                    self.exp_bias = nn.Parameter(torch.tensor(0.0))

    def _compute_step_input_dim(self):
        """Calculate step_input_dim based on enabled features - reusable for both step_mlp and temp_mlp"""
        step_input_dim = 0
        
        if self.use_step_ratio:
            step_input_dim += 1
        if self.use_last_3_nodes:
            step_input_dim += 3
        if self.use_visited_mean:
            step_input_dim += self.embedding_dim
        if self.use_unvisited_mean:
            step_input_dim += self.embedding_dim
        if self.use_sin_cos_time:
            step_input_dim += 2
        if self.use_linear_time:
            step_input_dim += 1
        if self.use_depot_distance:
            step_input_dim += 1
        if self.use_tour_length:
            step_input_dim += 1
        if self.use_mean_dist_unvisited:
            step_input_dim += 1
        
        return step_input_dim

    def _generate_xx_yy(self, graph_size, batch_size, device):
        """
        Generate xx and yy coordinate tensors dynamically based on actual graph_size.
        
        Args:
            graph_size: Actual size of the graph (from input)
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            xx: (batch_size, graph_size*graph_size) - source node indices
            yy: (batch_size, graph_size*graph_size) - destination node indices
        """
        # Create coordinate grids directly (compatible with all PyTorch versions)
        # xx: source nodes (repeated for each destination)
        # yy: destination nodes (tiled for each source)
        nodes = torch.arange(graph_size, device=device)
        
        # Create xx: [0,0,0,...,1,1,1,...,graph_size-1,graph_size-1,...]
        # Each source node repeated graph_size times
        xx_flat = nodes.repeat_interleave(graph_size)
        
        # Create yy: [0,1,2,...,0,1,2,...,0,1,2,...]
        # All destination nodes repeated graph_size times
        yy_flat = nodes.repeat(graph_size)
        
        # Expand to batch dimension
        xx = xx_flat.unsqueeze(0).expand(batch_size, -1)
        yy = yy_flat.unsqueeze(0).expand(batch_size, -1)
        
        return xx, yy

    def _get_step_features(self, state, embeddings, mat=None, input=None):
        """Extract step features for MLP input - granular feature control"""
        batch_size = state.ids.size(0)
        graph_size = embeddings.size(1)
        
        features_list = []
        
        # Step ratio (cheap, 1 feature)
        if self.use_step_ratio:
            step_ratio = (state.i.float() / graph_size).unsqueeze(0).expand(batch_size, 1)
            step_ratio = torch.where(torch.isnan(step_ratio), torch.zeros_like(step_ratio), step_ratio)
            features_list.append(step_ratio)
        
        # Last 3 visited nodes (cheap, 3 features)
        if self.use_last_3_nodes:
            last_3_nodes = torch.zeros(batch_size, 3, device=embeddings.device)
            if state.i.item() > 0:
                for i in range(min(3, state.i.item())):
                    if state.i.item() - i - 1 >= 0:
                        last_3_nodes[:, i] = 1.0
            features_list.append(last_3_nodes)
        
        # Mean of visited embeddings (EXPENSIVE: embedding_dim features)
        if self.use_visited_mean:
            visited_mask = state.visited_.squeeze(1).unsqueeze(-1)
            embeddings_clean = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)
            visited_embeddings = embeddings_clean * visited_mask.float()
            num_visited = torch.clamp(visited_mask.sum(1).float(), min=1.0)
            visited_mean = visited_embeddings.sum(1) / num_visited
            visited_mean = torch.where(torch.isnan(visited_mean), embeddings_clean.mean(1), visited_mean)
            features_list.append(visited_mean)
        
        # Time encoding: sin/cos (2 features) or linear (1 feature)
        if self.use_sin_cos_time:
            normalized_time = state.lengths
            sin_time = torch.sin(2 * math.pi * normalized_time)
            cos_time = torch.cos(2 * math.pi * normalized_time)
            features_list.append(sin_time)
            features_list.append(cos_time)
        elif self.use_linear_time:
            # Linear time encoding (1 feature)
            normalized_time = state.lengths
            # Ensure proper shape: (batch_size, 1)
            if normalized_time.dim() == 1:
                normalized_time = normalized_time.unsqueeze(1)
            elif normalized_time.dim() == 0:
                normalized_time = normalized_time.unsqueeze(0).unsqueeze(0)
            features_list.append(normalized_time)
                
        # Distance to depot (cheap, 1 feature)
        if self.use_depot_distance:
            _, ind = torch.max(input, dim=2)
            depot_distance = mat.__getd__(ind, state.prev_a, state.first_a, state.lengths)
            depot_distance = depot_distance.unsqueeze(1) if depot_distance.dim() == 1 else depot_distance
            features_list.append(depot_distance)
        
        # Tour length so far (cheap, 1 feature)
        if self.use_tour_length:
            tour_length = state.lengths / (graph_size * 0.1)
            features_list.append(tour_length)
        
        # Mean of unvisited embeddings (EXPENSIVE: embedding_dim features)
        if self.use_unvisited_mean:
            unvisited_mask = (~state.visited_).squeeze(1).unsqueeze(-1)
            embeddings_clean = torch.where(torch.isnan(embeddings), torch.zeros_like(embeddings), embeddings)
            unvisited_embeddings = embeddings_clean * unvisited_mask.float()
            num_unvisited = torch.clamp(unvisited_mask.sum(1).float(), min=1.0)
            unvisited_mean = unvisited_embeddings.sum(1) / num_unvisited
            unvisited_mean = torch.where(torch.isnan(unvisited_mean), embeddings_clean.mean(1), unvisited_mean)
            features_list.append(unvisited_mean)
        
        # Mean distance to unvisited nodes (moderate cost, 1 feature)
        if self.use_mean_dist_unvisited:
            mean_dist_unvisited = self._compute_mean_distance_to_unvisited(
                state, mat, input, embeddings.device
            )
            features_list.append(mean_dist_unvisited)
        
        # Concatenate all enabled features
        if features_list:
            step_features = torch.cat(features_list, dim=1)
        else:
            # No features enabled - return empty tensor
            step_features = torch.zeros(batch_size, 0, device=embeddings.device)
        
        # Final NaN check
        step_features = torch.where(torch.isnan(step_features), 
                                    torch.zeros_like(step_features), 
                                    step_features)
        
        return step_features

    def _get_log_p(self, fixed, state, mat, input, normalize=True):

        # Original query computation
        query = fixed.context_node_projected + \
            self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, input))
        
        # Initialize temp_adjustment (will be set if temp_mlp is enabled)
        temp_adjustment = 1.0
        
        # Step-MLP enhancement (if enabled)
        if self.use_step_mlp:
            step_features = self._get_step_features(state, fixed.node_embeddings, mat, input)
            
            # Safety check on step_features
            if torch.isnan(step_features).any():
                print("NaN in step_features, using zero context nudge")
                context_nudge = torch.zeros(step_features.size(0), 1, fixed.node_embeddings.size(-1), device=step_features.device)
            else:
                context_nudge = self.step_mlp(step_features).unsqueeze(1)
                
                # Safety check on context_nudge
                if torch.isnan(context_nudge).any():
                    print("NaN in context_nudge from step_mlp")
                    context_nudge = torch.zeros_like(context_nudge)
            
            query = query + context_nudge
            
            # Safety check on query
            if torch.isnan(query).any():
                print("NaN in query after adding context_nudge")
                query = torch.where(torch.isnan(query), torch.zeros_like(query), query)
        
        # Temperature adjustment (can work independently or with step_mlp)
        if self.use_temp_mlp:
            # Compute step_features if not already computed by step_mlp
            if not self.use_step_mlp:
                step_features = self._get_step_features(state, fixed.node_embeddings, mat, input)
            
            # Safety check on step_features
            if torch.isnan(step_features).any():
                print("NaN in step_features for temp_mlp, using default temperature")
                temp_adjustment = 1.0
            else:
                temp_adjustment = self.temp_mlp(step_features).squeeze(-1) * 2.0 + 0.5
                
                # Safety check on temp_adjustment
                if torch.isnan(temp_adjustment).any():
                    print("NaN in temp_adjustment from temp_mlp")
                    temp_adjustment = 1.0

        # Option C: Pre-attention MLP (processes query before attention)
        if self.use_decoder_mlp_pre and self.decoder_mlp_pre is not None:
            # Process query through MLP to get refined query
            refined_query = self.decoder_mlp_pre(query)
            
            # Safety check
            if torch.isnan(refined_query).any():
                print("NaN in refined_query from decoder_mlp_pre, using original query")
                refined_query = torch.where(torch.isnan(refined_query), query, refined_query)
            
            # Use residual connection (recommended) or replace
            query = query + refined_query  # Residual connection
            
            # Safety check on final query
            if torch.isnan(query).any():
                print("NaN in query after decoder_mlp_pre, fixing...")
                query = torch.where(torch.isnan(query), torch.zeros_like(query), query)
        
        
        # Compute attention with Cost-Aware Gating (new)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()
        
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask,
            embeddings=fixed.node_embeddings,
            state=state, mat=mat, input=input
        )
        
        # Safety check on log_p before normalization
        if torch.isnan(log_p).any():
            print("NaN in log_p before normalization")
            log_p = torch.where(torch.isnan(log_p), torch.full_like(log_p, -1e9), log_p)
        
        if normalize:
            # Use temp_adjustment if temp_mlp is enabled, otherwise use default temperature
            if self.use_temp_mlp:
                log_p = F.log_softmax(log_p / (self.temp * temp_adjustment.unsqueeze(-1)), dim=-1)
            else:
                log_p = F.log_softmax(log_p / self.temp, dim=-1)
        
        # Final safety check
        if torch.isnan(log_p).any():
            print("NaN in log_p after normalization")
            log_p = torch.where(torch.isnan(log_p), torch.full_like(log_p, -1e9), log_p)
        
        return log_p, mask
        
    def set_decode_type(self, decode_type, temp=None, beam_width=1):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp
        if decode_type == "beam":
            self.beam_width = beam_width

    def _slice_time_window(self, mat, start_time, W, batch_size):
        """
        Slice time window from full time series cubic spline coefficients.
        
        Args:
            mat: DistanceMatrix object containing mat.mat (cubic spline coefficients)
            start_time: Starting time in normalized form (0.0 to 1.0) or time bin index (0-11)
            W: Window size in time bins. 
               If W=-1, use all bins from start_time to end of day (forward window mode)
            batch_size: Batch size for replication
        
        Returns:
            z_windowed: Tensor of shape (batch_size, n_cities, n_cities * W)
                    containing windowed traffic coefficients
        
        Examples:
            Fixed window: start_time=0.25 (6:00 AM), W=3
            → Extracts time bins [3, 4, 5] (6:00 AM to 12:00 PM)
            
            Forward window: start_time=0.25 (6:00 AM), W=-1
            → Extracts time bins [3, 4, 5, 6, 7, 8, 9, 10, 11] (6:00 AM to midnight)
        """
        # Step 1: Convert normalized time to time bin index
        if isinstance(start_time, float) and start_time <= 1.0:
            # Normalized time (0.0 to 1.0)
            start_bin = int(torch.floor(torch.tensor(start_time * self.max_t)).item()) % self.max_t
        else:
            # Already a time bin index
            start_bin = int(start_time) % self.max_t
        
        # Handle forward window mode (W=-1 means use all bins from start_bin to end)
        if W == -1 or W == 0:
            # Forward window: use all remaining bins from start_bin to end of day
            W = self.max_t - start_bin
            print(f"[Time Slicing] Forward window mode: using {W} bins from bin {start_bin} to end (bins [{start_bin}:{self.max_t}])")
        
        end_bin = start_bin + W
        
        # Step 2: Reshape mat.mat from (n_c * n_c * max_t,) to (n_c, n_c, max_t)
        # mat.mat shape: (n_cities * n_cities * max_t,)
        # After reshape: (100, 100, 12) = (source, dest, time)
        mat_reshaped = mat.mat.view(self.n_cities, self.n_cities, self.max_t)
        
        # Step 3: Handle wraparound (when window crosses midnight)
        if end_bin > self.max_t:
            # Window crosses the 24-hour boundary
            # Example: start_bin=10, W=5 → end_bin=15
            # Extract [10, 11] from current day and [0, 1, 2] from next day
            part1 = mat_reshaped[:, :, start_bin:self.max_t]  # [start_bin : 12]
            part2 = mat_reshaped[:, :, 0:(end_bin - self.max_t)]  # [0 : end_bin-12]
            mat_windowed = torch.cat([part1, part2], dim=2)  # Concatenate along time dimension
            print(f"[Time Slicing] Wraparound: bins [{start_bin}:{self.max_t}] + [0:{end_bin - self.max_t}]")
        else:
            # Normal case: no wraparound
            # Example: start_bin=3, W=9 → extract bins [3, 4, 5, 6, 7, 8, 9, 10, 11]
            mat_windowed = mat_reshaped[:, :, start_bin:end_bin]
            print(f"[Time Slicing] Extracted window: bins [{start_bin}:{end_bin}], shape {mat_windowed.shape}")
        
        # mat_windowed shape: (100, 100, W)
        
        # Step 4: Reshape to encoder input format
        # Flatten destination and time dimensions: (100, 100, W) → (100, 100*W)
        # Then add batch dimension: (1, 100, 100*W)
        z_windowed = mat_windowed.contiguous().view(1, self.n_cities, self.n_cities * W)
        
        # Step 5: Repeat for batch
        z_windowed = z_windowed.repeat(batch_size, 1, 1)
        # Final shape: (batch_size, 100, 100*W)
        
        print(f"[Time Slicing] Extracted window: bins [{start_bin}:{end_bin%self.max_t}], "
            f"shape {z_windowed.shape}, features per node: {self.n_cities * W}")
        
        return z_windowed

    def _initialize_window_state(self, start_time, refresh_strategy='one_time', 
                                 refresh_interval=0.5, buffer_k_moves=2):
        """
        Initialize window state tracking for Safe Refresh mechanism.
        
        Args:
            start_time: Starting time (normalized 0.0-1.0 or bin index 0-11)
            refresh_strategy: Refresh strategy ('buffer', 'one_time', 'periodic', 'combined', 'none')
            refresh_interval: Refresh interval for periodic strategy (normalized time)
            buffer_k_moves: Number of moves to look ahead for buffer rule
        """
        # Convert start_time to bin index if needed
        if isinstance(start_time, float) and 0.0 <= start_time <= 1.0:
            # Normalized time: convert to bin index
            start_bin = int(torch.floor(torch.tensor(start_time * self.max_t)).item())
        else:
            # Already a bin index
            start_bin = int(start_time) if start_time is not None else 0
        
        # Calculate window end
        end_bin = start_bin + self.window_size_W
        
        # Store window state
        self.current_window_start_bin = start_bin
        self.current_window_end_bin = end_bin # store the window end bin/time as a bin index
        self.last_refresh_time = start_time if isinstance(start_time, float) else start_time / self.max_t
        self.refresh_count = 0
        
        # Store refresh parameters
        self.refresh_strategy = refresh_strategy
        self.refresh_interval = refresh_interval
        self.buffer_k_moves = buffer_k_moves
        
        print(f"[Window State] Initialized: start_bin={start_bin}, end_bin={end_bin}, "
              f"strategy={refresh_strategy}")

    def _should_refresh_encoder(self, current_time, state, mat):
        """
        Check if encoder should be refreshed based on current time and refresh strategy.
        
        Args:
            current_time: Current time (normalized 0.0-1.0 or bin index)
            state: Current decoder state
            mat: DistanceMatrix object
        
        Returns:
            bool: True if encoder should be refreshed
            str: Reason for refresh ('boundary', 'buffer', 'periodic', 'none')
        """
        if not self.use_time_slicing or self.refresh_strategy == 'none':
            return False, 'none'
        
        # Convert current_time to bin index if needed
        if isinstance(current_time, float) and 0.0 <= current_time <= 1.0:
            current_bin = int(torch.floor(torch.tensor(current_time * self.max_t)).item())
        else:
            current_bin = int(current_time) if current_time is not None else 0
        
        # Strategy 1: One-time refresh (when boundary is reached)
        if self.refresh_strategy in ['one_time', 'combined']:
            # Check if current time has reached or exceeded window end
            if current_bin >= self.current_window_end_bin:
                return True, 'boundary'
        
        # Strategy 2: Buffer rule (k-moves lookahead)
        if self.refresh_strategy in ['buffer', 'combined']:
            # Simulate k greedy moves and check if estimated time exceeds window
            # This is a simplified check - we'll estimate based on average travel time
            # TODO: Implement actual k-moves simulation
            estimated_future_bin = current_bin + self.buffer_k_moves  # Simplified
            if estimated_future_bin >= self.current_window_end_bin:
                return True, 'buffer'
        
        # Strategy 3: Periodic refresh
        if self.refresh_strategy in ['periodic', 'combined']:
            # Check if enough time has passed since last refresh
            if self.last_refresh_time is not None:
                time_since_refresh = current_time - self.last_refresh_time
                if time_since_refresh >= self.refresh_interval:
                    return True, 'periodic'
        
        return False, 'none'

    def _refresh_encoder(self, current_time, mat, input):
        """
        Refresh encoder with new time window starting from current time.
        
        Args:
            current_time: Current time (normalized 0.0-1.0 or bin index)
            mat: DistanceMatrix object
            input: Input node features
        
        Returns:
            embeddings: New node embeddings from refreshed encoder
        """
        print(f"[Refresh] Refreshing encoder at time={current_time}, refresh_count={self.refresh_count + 1}")
        
        # Update window state to start from current time
        self._initialize_window_state(
            start_time=current_time,
            refresh_strategy=self.refresh_strategy,
            refresh_interval=self.refresh_interval,
            buffer_k_moves=self.buffer_k_moves
        )
        
        # Re-encode with new window
        x = self._init_embed(input)  # Embed node features
        
        if self.use_time_slicing:
            # Calculate actual window size (handle forward window mode: W=-1)
            if self.window_size_W == -1:
                # Forward window: use all bins from current_time to end of day
                if isinstance(current_time, float) and current_time <= 1.0:
                    start_bin = int(torch.floor(torch.tensor(current_time * self.max_t)).item()) % self.max_t
                else:
                    start_bin = int(current_time) % self.max_t
                actual_W = self.max_t - start_bin
            else:
                actual_W = self.window_size_W
            
            # Extract new windowed coefficients
            z_windowed = self._slice_time_window(
                mat, 
                start_time=current_time,
                W=self.window_size_W,  # Pass original W (may be -1)
                batch_size=input.size(0)
            )
            
            # Get node indices (which nodes are in this problem instance)
            _, ind = torch.max(input, dim=2)
            # Shape: (batch_size, graph_size)
            
            # Calculate actual features per node
            actual_features = self.n_cities * actual_W
            
            # Gather coefficients for selected nodes
            tr = z_windowed.gather(
                1,  # Gather along dimension 1 (source nodes)
                ind.view(input.size(0), -1, 1).expand(
                    input.size(0),           # batch_size
                    input.size(1),           # graph_size
                    actual_features  # features: 100*actual_W
                )
            )
            # Shape: (batch_size, graph_size, 100*actual_W)
            
            # Pad to max size if needed (for fixed windows smaller than max_t)
            # The embedding layer expects (batch, graph_size, 100*12)
            max_features = self.n_cities * self.max_t
            if actual_features < max_features:
                # Pad with zeros to match embedding layer input size
                padding_size = max_features - actual_features
                padding = torch.zeros(
                    tr.size(0), 
                    tr.size(1), 
                    padding_size, 
                    device=tr.device, 
                    dtype=tr.dtype
                )
                tr_padded = torch.cat([tr, padding], dim=2)
            else:
                tr_padded = tr
            
            # Project to embedding dimension using windowed layer
            y = self.embed_windowed_traffic(tr_padded)
            # Shape: (batch_size, graph_size, embedding_dim)
        else:
            # Use full time series (original method)
            z = mat.mat.view(1, self.n_cities, self.n_cities * self.max_t).repeat(input.size(0), 1, 1)
            _, ind = torch.max(input, dim=2)
            tr = z.gather(
                1,
                ind.view(input.size(0), -1, 1).expand(
                    input.size(0), input.size(1), self.n_cities * self.max_t
                )
            )
            tr = tr.squeeze(2)
            y = self.embed_static_traffic(tr)
        
        # Concatenate position and traffic embeddings
        h = torch.cat([x, y], dim=-1)  # Shape: (batch_size, graph_size, 2 * embedding_dim)
        
        # Project to embedding dimension
        h = self.embed_static(h)  # Shape: (batch_size, graph_size, embedding_dim)
        
        # Encode through transformer
        embeddings, _ = self.embedder(h)  # Shape: (batch_size, graph_size, embedding_dim)
        
        self.refresh_count += 1
        return embeddings
    
    def forward(self, mat, input, return_pi=False, start_time=None):
        """
        Forward pass of the attention model.
        
        Args:
            mat: DistanceMatrix object with traffic coefficients
            input: (batch_size, graph_size, node_dim) input node features
            return_pi: Whether to return the output sequences
            start_time: Starting time for time slicing (0.0-1.0 normalized, or 0-11 bin index)
                    If None and use_time_slicing=True, defaults to 0.0 (midnight)
        
        Returns:
            cost: Tour costs
            ll: Log likelihoods
            pi: Tour sequences (if return_pi=True)
        """
        # Step 1: Embed node features (locations)
        x = self._init_embed(input)  # Shape: (batch, graph_size, embedding_dim)
        
        # Step 2: Process traffic patterns (TIME SLICING HAPPENS HERE)
        if self.use_time_slicing:
            # ============= WINDOWED TIME SLICING =============
            # Set default start time if not provided
            if start_time is None:
                start_time = 0.0  # Default to midnight
            
            # Calculate actual window size (handle forward window mode: W=-1)
            if self.window_size_W == -1:
                # Forward window: use all bins from start_time to end of day
                if isinstance(start_time, float) and start_time <= 1.0:
                    start_bin = int(torch.floor(torch.tensor(start_time * self.max_t)).item()) % self.max_t
                else:
                    start_bin = int(start_time) % self.max_t
                actual_W = self.max_t - start_bin
                print(f"\n[Time Slicing] Forward window mode: bins [{start_bin}:{self.max_t}] ({actual_W} bins from start to end)")
            else:
                actual_W = self.window_size_W
                print(f"\n[Time Slicing] Fixed window: W={actual_W}, start_time={start_time}")
            
            # Extract windowed coefficients
            z_windowed = self._slice_time_window(
                mat, 
                start_time, 
                self.window_size_W,  # Pass original W (may be -1 for forward window)
                input.size(0)  # batch_size
            )
            # Shape: (batch_size, 100, 100*actual_W)
            
            # Get node indices (which nodes are in this problem instance)
            _, ind = torch.max(input, dim=2)
            # Shape: (batch_size, graph_size)
            
            # Calculate actual features per node
            actual_features = self.n_cities * actual_W
            
            # Gather coefficients for selected nodes
            tr = z_windowed.gather(
                1,  # Gather along dimension 1 (source nodes)
                ind.view(input.size(0), -1, 1).expand(
                    input.size(0),           # batch_size
                    input.size(1),           # graph_size
                    actual_features  # features: 100*actual_W
                )
            )
            # Shape: (batch_size, graph_size, 100*actual_W)
            
            # Pad to max size if needed (for forward window that's smaller than max_t)
            # The embedding layer expects (batch, graph_size, 100*12)
            max_features = self.n_cities * self.max_t
            if actual_features < max_features:
                # Pad with zeros to match embedding layer input size
                padding_size = max_features - actual_features
                padding = torch.zeros(
                    tr.size(0), 
                    tr.size(1), 
                    padding_size, 
                    device=tr.device, 
                    dtype=tr.dtype
                )
                tr_padded = torch.cat([tr, padding], dim=2)
                print(f"[Time Slicing] Padded from {actual_features} to {max_features} features")
            else:
                tr_padded = tr
            
            # Project to embedding dimension using windowed layer
            y = self.embed_windowed_traffic(tr_padded)
            # Shape: (batch_size, graph_size, embedding_dim)
            
            print(f"[Time Slicing] Traffic features: {tr.shape} → {y.shape}")
        else:
            # ============= ORIGINAL FULL TIME SERIES =============
            print(f"\n[Original] Using full time series encoding (12 bins)")
            
            # Reshape and replicate full coefficient matrix
            n_c = self.n_cities
            max_t = self.max_t
            z = mat.mat.view(1, n_c, n_c * max_t).repeat(input.size(0), 1, 1)
            # Shape: (batch_size, n_cities, n_cities*max_t)
            
            # Get node indices
            _, ind = torch.max(input, dim=2)
            
            # Gather coefficients for selected nodes
            tr = z.gather(
                1,
                ind.view(input.size(0), -1, 1).expand(
                    input.size(0), input.size(1), n_c * max_t
                )
            )
            # Shape: (batch_size, graph_size, n_cities*max_t)
            
            # Project to embedding dimension using original layer
            y = self.embed_static_traffic(tr)
            # Shape: (batch_size, graph_size, embedding_dim)
        
        # Step 3: Combine node and traffic embeddings
        combined = torch.cat((x, y), dim=2)
        # Shape: (batch_size, graph_size, 2 * embedding_dim)
        
        # Step 4: Pass through encoder
        embeddings, _ = self.embedder(self.embed_static(combined))
        # Shape: (batch_size, graph_size, embedding_dim)
        
        self.embeddings = embeddings  # Store for later use
        
        #   Initialize window state for Safe Refresh (if time slicing enabled)
        if self.use_time_slicing and start_time is not None:
            # Get refresh parameters (these should be set from opts, but we'll use defaults if not set)
            refresh_strategy = getattr(self, 'refresh_strategy', None) or 'one_time'
            refresh_interval = getattr(self, 'refresh_interval', None) or 0.5
            buffer_k_moves = getattr(self, 'buffer_k_moves', None) or 2
            
            # Initialize window state tracking
            self._initialize_window_state(
                start_time=start_time,
                refresh_strategy=refresh_strategy,
                refresh_interval=refresh_interval,
                buffer_k_moves=buffer_k_moves
            )
        
        # Step 5: Decode (generate tour)
        _log_p, pi, state = self._inner(input, embeddings, mat)
        
        # Step 6: Calculate costs and log likelihoods
        cost, mask = get_costs(input, pi, state, mat)
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        
        if return_pi:
            return cost, ll, pi
        else:
            return cost, ll, None

    def _init_embed(self, x):
        return self.init_embed(x)
    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, embeddings, mat):
        """
        Inner decoding loop with Safe Refresh mechanism.
        Routes to beam search if decode_type is "beam".

        Args:
            input: Input node features
            embeddings: Initial node embeddings from encoder
            mat: DistanceMatrix object
        """
        # Check if beam search is requested
        if self.decode_type == "beam":
            return self._inner_beam(input, embeddings, mat, self.beam_width)

        # ===== DECODER LOOP =====
        outputs = []
        sequences = []

        state = StateTSP.initialize(input)
        state = state.addmask()

        batch_size = state.ids.size(0)

        # Perform decoding steps
        fixed = self._precompute(embeddings)
        
        while not (state.all_finished()):
            #   Check if encoder refresh is needed (before each decoding step)
            if self.use_time_slicing and self.refresh_strategy != 'none':
                # Get current time from state.lengths (normalized 0.0-1.0)
                # state.lengths shape: (batch_size, 1)
                if state.lengths.numel() > 0:
                    current_time = state.lengths[0, 0].item()  # Get first batch, first step
                else:
                    current_time = 0.0  # Default if empty
                
                # Check if refresh is needed : Pass currnet time to refersh check
                should_refresh, refresh_reason = self._should_refresh_encoder(
                    current_time=current_time,
                    state=state,
                    mat=mat
                )
                
                if should_refresh:
                    step_num = state.i.item() if state.i.numel() > 0 else 0
                    print(f"[Decoder] Refresh triggered: reason={refresh_reason}, "
                          f"current_time={current_time:.3f}, step={step_num}")
                    
                    # Refresh encoder with new window
                    new_embeddings = self._refresh_encoder(
                        current_time=current_time,
                        mat=mat,
                        input=input
                    )
                    
                    # Update embeddings
                    embeddings = new_embeddings
                    
                    # Recompute fixed attention data with new embeddings
                    fixed = self._precompute(embeddings)
                    
                    print(f"[Decoder] Encoder refreshed, continuing decoding...")

            # Original decoding step
            log_p, mask = self._get_log_p(fixed, state, mat, input)

            # Select the indices of the next nodes in the sequences
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            state = state.update(selected, mat, input)

            # Collect output of step
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1), state

    def _inner_beam(self, input, embeddings, mat, beam_width):
        """
        Fully batched beam search - processes all beams in parallel.
        
        Complexity per step: O(1) forward passes instead of O(batch_size * beam_width)
        
        Args:
            input: Input node features (batch_size, graph_size, input_dim)
            embeddings: Node embeddings (batch_size, graph_size, embed_dim)
            mat: DistanceMatrix object
            beam_width: Number of beams to maintain per instance
        
        Returns:
            outputs: Dummy tensor for compatibility (batch_size, graph_size-1, graph_size)
            sequences: Best sequences (batch_size, graph_size-1)
            final_state: Final StateTSP state
        """
        batch_size = input.size(0)
        graph_size = input.size(1)
        device = input.device
        embed_dim = embeddings.size(-1)
        
        # ========================================================================
        # STEP 1: Expand everything to (batch_size * beam_width) for parallel processing
        # ========================================================================
        
        # Expand input: (batch_size, graph_size, dim) -> (batch_size * beam_width, graph_size, dim)
        input_expanded = input.unsqueeze(1).expand(-1, beam_width, -1, -1)
        input_expanded = input_expanded.contiguous().view(batch_size * beam_width, graph_size, -1)
        
        # Expand embeddings similarly
        embeddings_expanded = embeddings.unsqueeze(1).expand(-1, beam_width, -1, -1)
        embeddings_expanded = embeddings_expanded.contiguous().view(batch_size * beam_width, graph_size, embed_dim)
        
        # Initialize state for all beams at once
        state = StateTSP.initialize(input_expanded)
        state = state.addmask()
        
        # Precompute fixed attention data for expanded batch
        fixed = self._precompute(embeddings_expanded)
        
        # ========================================================================
        # STEP 2: Initialize beam tracking tensors
        # ========================================================================
        
        # Cumulative log probabilities: (batch_size, beam_width)
        # First beam starts at 0, others at -inf (only first beam active initially)
        beam_log_probs = torch.zeros(batch_size, beam_width, device=device)
        beam_log_probs[:, 1:] = float('-inf')
        
        # Track sequences: (batch_size, beam_width, max_steps)
        sequences = torch.zeros(batch_size, beam_width, graph_size - 1, dtype=torch.long, device=device)
        
        # Track if beams are finished: (batch_size, beam_width)
        finished = torch.zeros(batch_size, beam_width, dtype=torch.bool, device=device)
        
        # ========================================================================
        # STEP 3: Main decoding loop
        # ========================================================================
        
        for step in range(graph_size - 1):
            # ------------------------------------------------------------------
            # 3a: Get log probabilities for ALL beams in ONE forward pass
            # ------------------------------------------------------------------
            log_p, mask = self._get_log_p(fixed, state, mat, input_expanded)
            # log_p: (batch_size * beam_width, 1, graph_size)
            # mask: (batch_size * beam_width, 1, graph_size)
            
            log_p = log_p.squeeze(1)  # (batch_size * beam_width, graph_size)
            mask = mask.squeeze(1)    # (batch_size * beam_width, graph_size)
            
            # Reshape to (batch_size, beam_width, graph_size)
            log_p = log_p.view(batch_size, beam_width, graph_size)
            mask = mask.view(batch_size, beam_width, graph_size)
            
            # Apply mask: visited nodes get -inf
            log_p = log_p.masked_fill(mask, float('-inf'))
            
            # ------------------------------------------------------------------
            # 3b: Compute scores for all (beam, next_node) combinations
            # ------------------------------------------------------------------
            # scores[b, k, n] = beam_log_probs[b, k] + log_p[b, k, n]
            # This gives the total score if we extend beam k with node n
            
            scores = beam_log_probs.unsqueeze(-1) + log_p  # (batch_size, beam_width, graph_size)
            
            # Handle finished beams: keep their score, don't expand
            # Set all expansion scores to -inf except a "stay" option
            finished_expanded = finished.unsqueeze(-1).expand_as(scores)
            scores = scores.masked_fill(finished_expanded, float('-inf'))
            
            # Flatten beam and node dimensions for topk selection
            scores_flat = scores.view(batch_size, -1)  # (batch_size, beam_width * graph_size)
            
            # ------------------------------------------------------------------
            # 3c: Select top beam_width scores using torch.topk (vectorized!)
            # ------------------------------------------------------------------
            top_scores, top_indices = torch.topk(
                scores_flat, 
                k=min(beam_width, scores_flat.size(-1)), 
                dim=-1,
                largest=True,
                sorted=True
            )
            # top_scores: (batch_size, beam_width)
            # top_indices: (batch_size, beam_width) - encodes beam_idx * graph_size + node_idx
            
            # Decode which beam and which node each top score came from
            parent_beam_indices = top_indices // graph_size  # (batch_size, beam_width)
            selected_nodes = top_indices % graph_size        # (batch_size, beam_width)
            
            # ------------------------------------------------------------------
            # 3d: Reorganize beams based on selection (all vectorized!)
            # ------------------------------------------------------------------
            
            # Update cumulative log probabilities
            beam_log_probs = top_scores
            
            # Reorder sequences according to which parent beams were selected
            # batch_indices: [[0,0,0,...], [1,1,1,...], ...]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, beam_width)
            
            # Gather previous sequences from selected parent beams
            sequences = sequences[batch_indices, parent_beam_indices]  # (batch_size, beam_width, max_steps)
            sequences[:, :, step] = selected_nodes  # Add newly selected nodes
            
            # Update finished status based on parent beams
            finished = finished[batch_indices, parent_beam_indices]
            
            # Mark beams as finished if they've reached the end
            finished = finished | (step == graph_size - 2)  # Finished after last step
            
            # ------------------------------------------------------------------
            # 3e: Update state for next iteration (batched tensor operations)
            # ------------------------------------------------------------------
            
            # Compute flat indices for gathering from (batch_size * beam_width) tensors
            # flat_idx[b, k] = b * beam_width + parent_beam_indices[b, k]
            flat_parent_indices = (batch_indices * beam_width + parent_beam_indices).view(-1)
            # flat_parent_indices: (batch_size * beam_width,)
            
            # Reorganize state tensors based on selected parent beams
            new_prev_a = state.prev_a[flat_parent_indices]
            new_visited = state.visited_[flat_parent_indices]
            new_lengths = state.lengths[flat_parent_indices]
            new_ids = state.ids[flat_parent_indices]
            new_first_a = state.first_a[flat_parent_indices]
            
            # Create reorganized state
            state = state._replace(
                prev_a=new_prev_a,
                visited_=new_visited,
                lengths=new_lengths,
                ids=new_ids,
                first_a=new_first_a
            )
            
            # Now update state with newly selected nodes
            selected_flat = selected_nodes.view(-1)  # (batch_size * beam_width,)
            state = state.update(selected_flat, mat, input_expanded)
        
        # ========================================================================
        # STEP 4: Extract best sequences from completed beams
        # ========================================================================
        
        # Find best beam for each instance
        best_beam_indices = beam_log_probs.argmax(dim=-1)  # (batch_size,)
        batch_indices_1d = torch.arange(batch_size, device=device)
        
        # Extract best sequences
        best_sequences = sequences[batch_indices_1d, best_beam_indices]  # (batch_size, graph_size-1)
        
        # Extract final state components for best beams
        flat_best_indices = batch_indices_1d * beam_width + best_beam_indices
        
        final_state = state._replace(
            loc=input,  # Original input (not expanded)
            prev_a=state.prev_a[flat_best_indices],
            visited_=state.visited_[flat_best_indices],
            lengths=state.lengths[flat_best_indices],
            ids=state.ids[flat_best_indices],
            first_a=state.first_a[flat_best_indices],
            i=state.i
        )
        
        # Create dummy outputs for API compatibility
        outputs = torch.zeros(batch_size, graph_size - 1, graph_size, device=device)
        
        return outputs, best_sequences, final_state

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
    def _get_log_p(self, fixed, state, mat, input, normalize=True):

        # Original query computation
        query = fixed.context_node_projected + \
            self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, mat, input))
        
        # Initialize temp_adjustment (will be set if temp_mlp is enabled)
        temp_adjustment = 1.0
        step_features = None
        
        # Step-MLP enhancement (if enabled)
        if self.use_step_mlp:
            step_features = self._get_step_features(state, fixed.node_embeddings, mat, input)
            context_nudge = self.step_mlp(step_features).unsqueeze(1)
            query = query + context_nudge
        
        # Temperature adjustment (can work independently or with step_mlp)
        if self.use_temp_mlp:
            # Compute step_features if not already computed by step_mlp
            if not self.use_step_mlp:
                step_features = self._get_step_features(state, fixed.node_embeddings, mat, input)
            
            if step_features is not None:
                temp_adjustment = self.temp_mlp(step_features).squeeze(-1) * 2.0 + 0.5
                # Safety check
                if torch.isnan(temp_adjustment).any():
                    temp_adjustment = 1.0
        
        # Compute attention
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()
        
        log_p, glimpse = self._one_to_many_logits(
            query, glimpse_K, glimpse_V, logit_K, mask,
            embeddings=fixed.node_embeddings,
            state=state, mat=mat, input=input
        )
        
        if normalize:
            # Handle temperature adjustment properly
            if self.use_temp_mlp:
                log_p = F.log_softmax(log_p / (self.temp * temp_adjustment.unsqueeze(-1)), dim=-1)
            else:
                log_p = F.log_softmax(log_p / self.temp, dim=-1)
        
        return log_p, mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask,embeddings=None, state=None, mat=None, input=None): # NEW - Added parameters for heuristic blending

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Apply decoder MLP if enabled (Option 1: Post-attention MLP)
        if self.use_decoder_mlp and self.decoder_mlp is not None:
            # Process glimpse through MLP to get refined representation
            refined_glimpse = self.decoder_mlp(glimpse)
            final_Q = refined_glimpse
        else:
            # Baseline: Use glimpse directly without MLP refinement
            final_Q = glimpse

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        #final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # Check if logits already has NaN BEFORE cost-aware gating
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("NaN/Inf in base logits BEFORE cost-aware gating")
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

        #   Add Cost-Aware Gating (Soft Bias)
        if self.use_cost_aware_gating and embeddings is not None:
            try:
                # Compute heuristic logits
                heuristic_logits = self.heuristic_computer.compute_heuristic_logits(
                    embeddings, state, mat, input
                )  # Shape: (batch_size, graph_size)
                
                # Ensure heuristic_logits is finite
                heuristic_logits = torch.where(torch.isfinite(heuristic_logits), heuristic_logits, torch.zeros_like(heuristic_logits))
                
                # Expand to match logits shape
                heuristic_logits = heuristic_logits.unsqueeze(1).expand(-1, num_steps, -1)
                
                # Apply nonlinear transformation if enabled
                if hasattr(self, 'transform'):
                    if self.transform_type == 'piecewise':
                        heuristic_logits = self.transform(heuristic_logits.unsqueeze(-1)).squeeze(-1)
                    elif self.transform_type == 'exponential':
                        heuristic_logits = self.exp_scale * torch.exp(heuristic_logits + self.exp_bias)
                
                # Final safety check after transformation
                heuristic_logits = torch.where(torch.isfinite(heuristic_logits), heuristic_logits, torch.zeros_like(heuristic_logits))
                
                # Add heuristic bias
                lambda_clipped = torch.clamp(self.lambda_heuristic, min=0.0, max=2.0)
                logits = logits + lambda_clipped * heuristic_logits
                
                # Safety check after adding heuristic
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("NaN/Inf in logits after cost-aware gating, fixing...")
                    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
                    
            except Exception as e:
                print(f"Error in cost-aware gating: {e}")

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
            
        # Final safety: replace -inf with very negative number to avoid NaN in softmax
        logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9))

        return logits, glimpse.squeeze(-2)
        
    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        elif self.decode_type == "beam":
            # Beam search handles selection internally, this shouldn't be called
            # But if it is, fall back to greedy
            _, selected = probs.max(1)
    
        else:
            assert False, f"Unknown decode type: {self.decode_type}"
        return selected
    def _get_parallel_step_context(self, embeddings, state, mat, input):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """
        b_s, i_s = embeddings.size(0), embeddings.size(1)
        _, ind = torch.max(input, dim=2)

        #print(f"DEBUG _make_heads: b_s={b_s}, i_s={i_s}")
        #print(f"DEBUG _make_heads: self.xx.shape={self.xx.shape}")
        #print(f"DEBUG _make_heads: self.yy.shape={self.yy.shape}")
        #xx_repeated = self.xx.repeat(b_s, 1, 1).view(b_s, i_s*i_s)
        #print(f"DEBUG _make_heads: xx_repeated.shape={xx_repeated.shape}")
        #yy_repeated = self.yy.repeat(b_s, 1, 1).view(b_s, i_s*i_s)
        #print(f"DEBUG _make_heads: yy_repeated.shape={yy_repeated.shape}")
        # Generate xx and yy dynamically based on actual graph_size
        xx_repeated, yy_repeated = self._generate_xx_yy(i_s, b_s, embeddings.device)
        #print(f"DEBUG _make_heads: ind.shape={ind.shape}")
        getddd_result = mat.__getddd__(ind, xx_repeated, yy_repeated, state.lengths)
        #print(f"DEBUG _make_heads: getddd_result.shape={getddd_result.shape}")
        reshaped = getddd_result.view(b_s, 1, i_s*i_s)
        #print(f"DEBUG _make_heads: reshaped.shape={reshaped.shape}")

        # ADD THESE NEW DEBUG LINES:
        #print(f"DEBUG: self.project_traffic.in_features = {self.project_traffic.in_features}")
        #print(f"DEBUG: self.project_traffic.weight.shape = {self.project_traffic.weight.shape}")
        #print(f"DEBUG: reshaped.view(b_s, -1).shape = {reshaped.view(b_s, -1).shape}")

        current_traffic = self.project_traffic(reshaped)

        #current_traffic = self.project_traffic(mat.__getddd__(ind, xx_repeated, yy_repeated, state.lengths).view(b_s, 1, i_s*i_s))

        #current_traffic = self.project_traffic(mat.__getddd__(ind, self.xx.repeat(b_s, 1, 1).view(b_s, i_s*i_s), self.yy.repeat(b_s, 1, 1).view(b_s, i_s*i_s), state.lengths).view(b_s, 1, i_s*i_s))
        # ADD DEBUG FOR project_visit:
        #print(f"DEBUG: self.project_visit.in_features = {self.project_visit.in_features}")
        #print(f"DEBUG: self.project_visit.weight.shape = {self.project_visit.weight.shape}")
        #print(f"DEBUG: state.visited_.shape = {state.visited_.shape}")
        #print(f"DEBUG: state.visited_.float().shape = {state.visited_.float().shape}")
        
        current_visit = self.project_visit(state.visited_.float())
        ss = embeddings.gather(1, torch.cat((state.first_a, state.prev_a), 1)[:, :, None].expand(b_s, 2, embeddings.size(-1)))
        #print(f"DEBUG: ss.view(b_s, 1, -1).shape = {ss.view(b_s, 1, -1).shape}")
        #print(f"DEBUG: current_traffic.shape = {current_traffic.shape}")
        #print(f"DEBUG: current_visit.shape = {current_visit.shape}")
        
        return torch.cat((ss.view(b_s, 1, -1), current_traffic, current_visit), dim=2)
        
    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _compute_mean_distance_to_unvisited(self, state, mat, input, device):
        """Compute mean travel time to all unvisited nodes - OPTIMIZED VECTORIZED"""
        batch_size = state.prev_a.size(0)
        _, ind = torch.max(input, dim=2)
        graph_size = ind.size(1)
        
        visited_mask = state.visited_.squeeze(1)  # (batch_size, graph_size)
        
        # Ensure state.lengths is the right shape
        # state.lengths is (batch_size, 1), we need it expandable to (batch_size, graph_size)
        lengths_expanded = state.lengths.expand(batch_size, graph_size)  # (batch_size, graph_size)
        
        # Vectorized computation: all distances at once
        # current_pos_expanded: repeat current position for all destinations
        current_pos = state.prev_a  # (batch_size, 1)
        current_pos_expanded = current_pos.expand(batch_size, graph_size)  # (batch_size, graph_size)
        
        # all_dest: all nodes [0, 1, 2, ..., graph_size-1]
        all_dest = torch.arange(graph_size, device=device).unsqueeze(0).expand(batch_size, graph_size)  # (batch_size, graph_size)
        
        # Compute all distances in one call
        all_distances = mat.__getddd__(
            ind,
            current_pos_expanded,
            all_dest,
            lengths_expanded
        )  # (batch_size, graph_size)
        
        # Mask visited nodes (1.0 for unvisited, 0.0 for visited)
        unvisited_mask = (1.0 - visited_mask.float())  # (batch_size, graph_size)
        
        # Compute mean only over unvisited nodes
        total_unvisited = (all_distances * unvisited_mask).sum(dim=1, keepdim=True)  # (batch_size, 1)
        num_unvisited = unvisited_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch_size, 1)
        mean_distance = total_unvisited / num_unvisited  # (batch_size, 1)
        
        return mean_distance