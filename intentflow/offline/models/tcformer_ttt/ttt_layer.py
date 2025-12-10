from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig

# Fallback for transformers versions where is_causal_conv1d_available is not available in import_utils
try:
    from transformers.utils.import_utils import is_causal_conv1d_available
except ImportError:
    def is_causal_conv1d_available():
        try:
            import causal_conv1d
            return True
        except ImportError:
            return False

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

class TTTConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        rope_theta=10000.0,
        use_gate=False,
        share_qk=False,
        ttt_layer_type="linear",
        ttt_base_lr=1.0,
        mini_batch_size=16,
        pre_conv=False,
        conv_kernel=4,
        scan_checkpoint_group_size=0,
        use_dual_form=True,
        learnable_init_state=False,
        ttt_reg_lambda=0.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.use_gate = use_gate
        self.share_qk = share_qk
        self.ttt_layer_type = ttt_layer_type
        self.ttt_base_lr = ttt_base_lr
        self.mini_batch_size = mini_batch_size
        self.pre_conv = pre_conv
        self.conv_kernel = conv_kernel
        self.scan_checkpoint_group_size = scan_checkpoint_group_size
        self.use_dual_form = use_dual_form
        self.learnable_init_state = learnable_init_state
        self.ttt_reg_lambda = ttt_reg_lambda
        
        # Add intermediate_size for compatibility
        self.intermediate_size = kwargs.get("intermediate_size", hidden_size * 4)
        self.pretraining_tp = kwargs.get("pretraining_tp", 1)

        super().__init__(**kwargs)


########################
### Backbone Modules ###
########################

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGluMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu # Hardcoded to silu/swish for simplicity

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Conv(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            bias=True,
            kernel_size=config.conv_kernel,
            groups=config.hidden_size,
            padding=config.conv_kernel - 1,
        )

    def __call__(self, hidden_states, cache_params=None):
        seq_len = hidden_states.shape[1]
        hidden_states = self.norm(hidden_states)
        # [B, C, L]
        hidden_states = hidden_states.transpose(1, 2)
        
        if causal_conv1d_fn is None:
            if cache_params is not None:
                if cache_params.seqlen_offset > 0:
                    conv_state = cache_params.conv_states_dic["pre_conv"][self.layer_idx]
                    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                    conv_state[:, :, -1] = hidden_states[:, :, 0]
                    cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_state)
                    hidden_states = torch.sum(conv_state * self.conv.weight[:, 0, :], dim=-1)
                    hidden_states += self.conv.bias
                    hidden_states = hidden_states.unsqueeze(-1)
                else:
                    conv_state = nn.functional.pad(
                        hidden_states,
                        (self.config.conv_kernel - hidden_states.shape[-1], 0),
                    )
                    cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_state)
                    hidden_states = self.conv(hidden_states)[..., :seq_len]
            else:
                hidden_states = self.conv(hidden_states)[..., :seq_len]
        else:
            conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states_dic["pre_conv"][self.layer_idx],
                    conv_weights,
                    self.conv.bias,
                    None,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states,
                        (self.config.conv_kernel - hidden_states.shape[-1], 0),
                    )
                    cache_params.conv_states_dic["pre_conv"][self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv.bias, activation=None)

        # [B, L, C]
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states

#########################
### TTT Layer Modules ###
#########################

def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out

def ln_fwd(x, gamma, beta, eps=1e-6):
    """Standard LayerNorm forward pass."""
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta
    return y

def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z

class TTTCache:
    """Minimal TTTCache."""
    def __init__(self, model, batch_size: int):
        self.seqlen_offset = 0
        # Placeholder for full implementation if needed
        self.ttt_params_dict = defaultdict(dict)
        self.conv_states_dic = defaultdict(dict)

class TTTBase(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.width = config.hidden_size
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.width // self.num_heads
        self.mini_batch_size = config.mini_batch_size

        # token_idx is a scale factor that scale the summation in Eqn. 4
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx, persistent=False)
        # make the scale factor learnable
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size,)))

        self.share_qk = config.share_qk
        self.conv_kernel = config.conv_kernel
        self._init_qkvo_proj()
        self._init_rope()
        # Learnable eta in Sec. 2.7
        self._init_ttt_lr_gate()
        self._init_ttt_ln()

        # use gating as in Mamba backbone
        self.use_gate = config.use_gate
        if self.use_gate:
            self.g_proj = nn.Linear(self.width, self.width, bias=False)

        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        if not self.share_qk:
            self.k_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)

        if self.share_qk:
            self.conv_q = nn.Conv1d(self.hidden_size, self.hidden_size, bias=True, kernel_size=self.conv_kernel, groups=self.hidden_size, padding=self.conv_kernel - 1)
            self.conv_k = nn.Conv1d(self.hidden_size, self.hidden_size, bias=True, kernel_size=self.conv_kernel, groups=self.hidden_size, padding=self.conv_kernel - 1)

    def _init_rope(self):
        self.rope_theta = self.config.rope_theta
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )

    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states, cache_params: Optional[TTTCache] = None):
        if self.share_qk:
            xq, XV = self.q_proj(hidden_states), self.v_proj(hidden_states)
            seq_len = xq.shape[1]
            xq = xq.transpose(1, 2)
            if causal_conv1d_fn is None:
                # Simplified convolution logic if causal_conv1d not available
                XQ = self.conv_q(xq)[..., :seq_len]
                XK = self.conv_k(xq)[..., :seq_len]
            else:
                # Implement proper call to causal_conv1d_fn if available
                # For now sticking to torch.conv1d for compatibility
                XQ = self.conv_q(xq)[..., :seq_len]
                XK = self.conv_k(xq)[..., :seq_len]
                
            XQ = XQ.transpose(1, 2)
            XK = XK.transpose(1, 2)
        else:
            XQ, XK, XV = (
                self.q_proj(hidden_states),
                self.k_proj(hidden_states),
                self.v_proj(hidden_states),
            )
        return XQ, XK, XV

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.config.ttt_base_lr * ttt_lr / self.head_dim

        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]
        token_idx = torch.clamp_min(token_idx, 0.0)

        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )
        return token_eta, ttt_lr_eta

    def apply_gate(self, hidden_states, ttt_output):
        y = self.g_proj(hidden_states)
        y = F.gelu(y, approximate="tanh")
        output = y * ttt_output
        return output

    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.width)

        XQ = XQ.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, L // mini_batch_size, mini_batch_size, self.head_dim)

        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * ttt_lr_eta
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def ttt(self, inputs, mini_batch_size, last_mini_batch_params_dict, cache_params: Optional[TTTCache] = None):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        B, L = hidden_states.shape[:2]
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None

        XQ, XK, XV = self.get_qkv_projections(hidden_states, cache_params=cache_params)

        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)
        
        # Simplified without permute_qk for now as we don't have exact jax layout requirement
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)

        output_hidden_states = []
        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod, last_mini_batch_params_dict = self.ttt(
                self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.ttt(
                self.get_ttt_inputs(inputs, reminder_len, cache_params),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        if self.use_gate:
            output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states

class TTTLinear(TTTBase):
    def __init__(self, config: TTTConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        # TTT model initialization for TTT-Linear
        if config.learnable_init_state:
             self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
             self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        else:
             # Fixed initialization (zeros or random) if not learnable
             self.register_buffer("W1", torch.zeros(self.num_heads, self.head_dim, self.head_dim))
             self.register_buffer("b1", torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params: Optional[TTTCache] = None,
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # Enforce dual form usage as per requirement
        use_dual_form = self.config.use_dual_form

        def compute_mini_batch(params_dict, inputs):
            W1_init = params_dict["W1_states"]
            b1_init = params_dict["b1_states"]

            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            eta_mini_batch = inputs["eta"]
            
            # --- Strategy 2: Regularized TTT ---
            # Penalize deviation from initial weights: W_new = W - eta * lambda * (W - W_init)
            # Effectively: W = (1 - alpha) * W + alpha * W_init
            reg_lambda = self.config.ttt_reg_lambda
            if reg_lambda > 0.0:
                # Use base_lr for regularization step size to avoid instability
                alpha = self.config.ttt_base_lr * reg_lambda
                # Apply regularization (pull towards self.W1 / self.b1)
                W1_init = (1.0 - alpha) * W1_init + alpha * self.W1.unsqueeze(0)
                b1_init = (1.0 - alpha) * b1_init + alpha * self.b1.unsqueeze(0)
            # -----------------------------------

            X1 = XK_mini_batch
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                # Dual form implementation
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                
                grad_W1_last = torch.zeros_like(W1_last) # Placeholder
                grad_b1_last = torch.zeros_like(b1_last) # Placeholder
            else:
                # Primal form (fallback/not used if dual form enforced)
                raise NotImplementedError("Primal form not implemented in this streamlined version")

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)
            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            self.config.scan_checkpoint_group_size if self.training else 0,
        )

        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict

class TTTBlock(nn.Module):
    def __init__(self, config: TTTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pre_conv = config.pre_conv

        if config.ttt_layer_type == "linear":
            ttt_layer = TTTLinear
        else:
            raise ValueError(f"Invalid ttt_layer_type: {config.ttt_layer_type}")

        self.seq_modeling_block = ttt_layer(config=config, layer_idx=layer_idx)

        self.mlp = SwiGluMLP(config)
        if self.pre_conv:
            self.conv = Conv(config, layer_idx)

        self.seq_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        if self.pre_conv:
            residual = hidden_states
            hidden_states = self.conv(hidden_states, cache_params=cache_params)
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.seq_norm(hidden_states)

        hidden_states = self.seq_modeling_block(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_params=cache_params,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
