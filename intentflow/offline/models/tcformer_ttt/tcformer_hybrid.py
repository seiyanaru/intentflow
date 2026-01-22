import torch
import torch.nn as nn
from einops import rearrange
from models.tcformer.tcformer import MultiKernelConvBlock, TCNHead, ClassificationModule
from models.tcformer.tcformer import _GQAttention, DropPath, _build_rotary_cache 
from models.tcformer_ttt.ttt_layer import TTTConfig, TTTLinear, TTTCache
import torch.nn.functional as F

class TTTAdapter(nn.Module):
    """
    TTTを用いた軽量アダプターモジュール
    Structure: DownProj -> TTT-Linear -> UpProj
    ボトルネック構造により計算コストを削減しつつ適応を行う。
    """
    def __init__(self, config: TTTConfig, input_dim: int, adapter_dim: int, layer_idx: int = 0):
        super().__init__()
        
        # TTT Core: Inner Loop Optimization happens here
        # Configをコピーしてアダプター用に次元数を調整
        adapter_config = TTTConfig(**config.to_dict())
        
        # Head数の決定 (最低1)
        adapter_heads = max(1, config.num_attention_heads // 4)
        adapter_config.num_attention_heads = adapter_heads
        
        # adapter_dim を num_heads の倍数に丸める (切り捨て)
        adapter_dim = (adapter_dim // adapter_heads) * adapter_heads
        if adapter_dim == 0:
            adapter_dim = adapter_heads # 最低限確保
            
        adapter_config.hidden_size = adapter_dim
        # 中間層サイズも調整
        adapter_config.intermediate_size = adapter_dim * 4
        
        # Projections (補正後のadapter_dimを使用)
        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(adapter_dim, input_dim)
        
        # TTT Layer (Dual Form)
        self.ttt_layer = TTTLinear(adapter_config, layer_idx=layer_idx)
        self.norm = nn.LayerNorm(adapter_dim)
        
        # Zero-Init: 学習初期はアダプターの影響をゼロにする（安定性確保のため）
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, lr_scale: torch.Tensor | None = None, cache_params: TTTCache | None = None):
        # x: [B, T, D]
        # 1. 圧縮 (Down-projection)
        x = self.down_proj(x)
        x = self.act(x)
        
        # 2. 適応 (TTT: Test-Time Training)
        # ここで入力データに応じた重み更新が行われる
        x = self.ttt_layer(x, lr_scale=lr_scale, cache_params=cache_params)
        
        # 3. 復元 (Up-projection)
        x = self.norm(x)
        x = self.up_proj(x)
        
        return x

class GatingModule(nn.Module):
    """
    Input-dependent dynamic gating mechanism.
    Computes alpha based on global statistics of the input features.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 16, init_bias: float = -4.0):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Initialize bias to output a small value (close to 0) initially
        nn.init.constant_(self.net[2].bias, init_bias)
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[2].weight)

    def forward(self, x, extra: torch.Tensor | None = None):
        # x: [B, T, D]
        # Compute global statistics over time (mean + std)
        x_mean = x.mean(dim=1)                      # [B, D]
        x_std = x.std(dim=1, unbiased=False)        # [B, D]
        if extra is None:
            gate_in = torch.cat([x_mean, x_std], dim=-1)  # [B, 2D]
        else:
            # extra: [B, 1] (e.g., attention-path variability proxy)
            gate_in = torch.cat([x_mean, x_std, extra], dim=-1)  # [B, 2D+1]
        gate_in = self.in_norm(gate_in)
        gate = self.net(gate_in) # [B, 1]
        return gate.unsqueeze(1) # [B, 1, 1] for broadcasting


class EntropyGating(nn.Module):
    """
    Entropy-driven gating: alpha = sigmoid(w * H + b).
    This is deliberately simple and *reactive*: it uses model uncertainty rather than feature statistics.
    """
    def __init__(
        self,
        init_w: float = 2.0,
        init_b: float = -3.0,
        threshold: float = 0.0,
        max_out: float = 1.0,
        hard_off_below_threshold: bool = True,
    ):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(init_w)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.register_buffer("threshold", torch.tensor(float(threshold)), persistent=False)
        self.register_buffer("max_out", torch.tensor(float(max_out)), persistent=False)
        self.hard_off_below_threshold = bool(hard_off_below_threshold)

    def forward(self, entropy_1d: torch.Tensor) -> torch.Tensor:
        # entropy_1d: [B]
        # Dead-zone: below threshold, force 0 (protect stable subjects like S1/S5).
        thr = self.threshold.to(device=entropy_1d.device, dtype=entropy_1d.dtype)
        h = entropy_1d - thr
        if self.hard_off_below_threshold:
            mask = (h > 0).to(entropy_1d.dtype)
            h = torch.clamp(h, min=0.0)
        else:
            mask = 1.0

        z = self.w * h + self.b
        out = torch.sigmoid(z) * mask
        max_out = self.max_out.to(device=entropy_1d.device, dtype=entropy_1d.dtype)
        return out * max_out  # [B]

class HybridBlock(nn.Module):
    """
    Attention (Fixed) + TTT Adapter (Adaptive) の並列ハイブリッドブロック
    """
    def __init__(
        self,
        d_model: int,
        q_heads: int,
        kv_heads: int,
        ttt_config: TTTConfig,
        adapter_ratio=0.25,
        dropout=0.4,
        drop_path_rate=0.0,
        use_dynamic_gating=False,
        layer_idx: int = 0,
        ttt_drop_prob: float = 0.0,  # Probability to drop TTT during training
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ttt_drop_prob = ttt_drop_prob
        self.norm1 = nn.LayerNorm(d_model)
        
        # --- Main Path: Fixed Self-Attention (普遍的な特徴抽出) ---
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        
        # --- Adapter Path: TTT (個人差への適応) ---
        # adapter_ratio (例: 0.25) に圧縮して計算コストを下げる
        adapter_dim = int(d_model * adapter_ratio)
        self.ttt_adapter = TTTAdapter(ttt_config, d_model, adapter_dim, layer_idx=layer_idx)
        
        # Gating Mechanism
        self.use_dynamic_gating = use_dynamic_gating
        if self.use_dynamic_gating:
            # mean(x) + std(x) + attn_std proxy
            self.gate = GatingModule(input_dim=2 * d_model + 1)
        # Always define static scale for compatibility (used only when dynamic gating is off)
        self.adapter_scale = nn.Parameter(torch.tensor(0.0))

        # Debug: last alpha per sample (set in forward)
        self.last_alpha = None

        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP Part (Shared)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, gate_alpha: torch.Tensor | None = None, enable_ttt: bool = True, lr_scale: torch.Tensor | None = None, cache_params: TTTCache | None = None):
        # Pre-Norm
        normed_x = self.norm1(x)
        
        # Parallel Execution (並列実行)
        
        # 1. Main Path: Attention (Context)
        # 既存の重みで安定した特徴を抽出
        out_attn = self.attn(normed_x, cos, sin)
        
        # 2. Adapter Path: TTT (Adaptation)
        # IMPORTANT: when enable_ttt=False (reactive 1st pass), do NOT run TTT at all.
        # TTT layers can update internal state; even alpha=0 would still adapt if we executed the adapter.
        
        # TTT Drop: During training, randomly drop TTT with probability ttt_drop_prob
        # This forces the model to also learn Attention-only representations
        should_drop_ttt = False
        if self.training and self.ttt_drop_prob > 0.0:
            should_drop_ttt = torch.rand(1).item() < self.ttt_drop_prob
        
        if enable_ttt and not should_drop_ttt:
            out_ttt = self.ttt_adapter(normed_x, lr_scale=lr_scale, cache_params=cache_params)
        else:
            out_ttt = None
        
        # Integration (統合)
        # X_new = X + Attention(X) + alpha * TTT(X)
        if out_ttt is None:
            # Attention-only pass (for entropy computation or TTT drop)
            self.last_alpha = torch.zeros(x.shape[0], device=x.device)
            x = x + self.drop_path(out_attn)
        else:
            # Adaptation pass
            if gate_alpha is not None:
                # gate_alpha: [B,1,1]
                alpha = gate_alpha
            elif self.use_dynamic_gating:
                # Legacy: feature-statistics gating (kept for compatibility)
                attn_std = out_attn.std(dim=1, unbiased=False).mean(dim=-1, keepdim=True)  # [B,1]
                alpha = self.gate(x, extra=attn_std)  # [B,1,1]
            else:
                alpha = self.adapter_scale.view(1, 1, 1)

            self.last_alpha = alpha.squeeze(-1).squeeze(-1).detach()  # [B]
            x = x + self.drop_path(out_attn + alpha * out_ttt)
        
        # MLP Block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class HybridEncoder(nn.Module):
    def __init__(self, d_model, trans_depth, q_heads, kv_heads, ttt_config, adapter_ratio=0.25, drop_path_max=0.25, use_dynamic_gating=False, ttt_drop_prob=0.0):
        super().__init__()
        self.d_model = d_model
        self.trans_depth = trans_depth
        
        # RoPE Cache (位置エンコーディング)
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self.head_dim = d_model // q_heads
        # Stochastic Depth rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_max, trans_depth)]
        
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=d_model,
                q_heads=q_heads,
                kv_heads=kv_heads,
                ttt_config=ttt_config,
                adapter_ratio=adapter_ratio, 
                drop_path_rate=dpr[i],
                use_dynamic_gating=use_dynamic_gating,
                layer_idx=i,
                ttt_drop_prob=ttt_drop_prob,  # Pass TTT drop probability
            )
            for i in range(trans_depth)
        ])

    def _rotary_cache(self, seq_len: int, device: torch.device):
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(self.head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

    def forward(self, x, gate_alpha: torch.Tensor | None = None, enable_ttt: bool = True, lr_scale: torch.Tensor | None = None, cache_params: TTTCache | None = None):
        # x: [B, T, D]
        seq_len = x.shape[1]
        cos, sin = self._rotary_cache(seq_len, x.device)
        
        alphas = []
        for layer in self.layers:
            x = layer(x, cos[:seq_len], sin[:seq_len], gate_alpha=gate_alpha, enable_ttt=enable_ttt, lr_scale=lr_scale, cache_params=cache_params)
            if getattr(layer, "last_alpha", None) is not None:
                alphas.append(layer.last_alpha)
        # Debug: mean alpha across layers (per sample)
        if alphas:
            self.last_alpha = torch.stack(alphas, dim=0).mean(dim=0).detach()
        else:
            self.last_alpha = None
        return x

class TCFormerHybridModule(nn.Module):
    def __init__(
        self,
        n_classes,
        Chans=22,
        F1=32,
        D=2,
        d_group=16,
        temp_kernel_lengths=[20, 32, 64],
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        dropout_clf=0.5,
        ttt_config=None,
        q_heads=4,
        kv_heads=4,
        trans_depth=2,
        adapter_ratio=0.25,
        use_dynamic_gating=False,
        use_group_attn=True,
        gating_mode: str = "feature_stats",  # "feature_stats" | "entropy"
        entropy_alpha_init_w: float = 2.0,
        entropy_alpha_init_b: float = -3.0,
        entropy_lr_init_w: float = 2.0,
        entropy_lr_init_b: float = -2.0,
        entropy_threshold: float = 0.95,
        alpha_max: float = 0.5,
        lr_scale_max: float = 0.5,
        entropy_gating_in_train: bool = False,
        ttt_drop_prob: float = 0.0,  # Probability to drop TTT during training (prevents TTT-dependency)
        pmax_threshold: float = 1.0,  # Only apply TTT when pmax < this value (1.0 = no filtering)
    ):
        super().__init__()
        
        # 1. MultiKernelConvBlock (Feature Extraction - 共通)
        self.conv_block = MultiKernelConvBlock(
            n_channels=Chans, 
            temp_kernel_lengths=temp_kernel_lengths,
            F1=F1, 
            D=D, 
            pool_length_1=pool_length_1, 
            pool_length_2=pool_length_2, 
            dropout=dropout_conv,
            d_group=d_group,
            # IMPORTANT: respect config; this can materially affect left/right separation
            use_group_attn=use_group_attn,
        )
        
        # Calculate feature dimensions
        n_groups = len(temp_kernel_lengths)
        self.F2 = d_group * n_groups
        
        # 2. Hybrid Encoder (Attention + TTT Adapter)
        if ttt_config is None: ttt_config = {}
        
        # TTTの設定 (デフォルト値を安全側に設定)
        self.ttt_cfg = TTTConfig(
            hidden_size=self.F2, 
            num_hidden_layers=1, # Adapter内で1層として使う
            num_attention_heads=q_heads,
            ttt_layer_type=ttt_config.get("layer_type", "linear"),
            ttt_base_lr=ttt_config.get("base_lr", 0.1), # マイルドな学習率
            mini_batch_size=ttt_config.get("mini_batch_size", 16),
            use_dual_form=True, # 高速化のため必須
            learnable_init_state=ttt_config.get("learnable_init_state", False),
            ttt_reg_lambda=ttt_config.get("ttt_reg_lambda", 0.01), # 正則化
            ttt_anchor_scale_mode=ttt_config.get("ttt_anchor_scale_mode", "none"),
            ttt_loss_scale=ttt_config.get("ttt_loss_scale", 1.0),
            ttt_grad_clip=ttt_config.get("ttt_grad_clip", 1.0) # Gradient clipping
        )
        
        self.hybrid_encoder = HybridEncoder(
            d_model=self.F2,
            trans_depth=trans_depth,
            q_heads=q_heads,
            kv_heads=kv_heads,
            ttt_config=self.ttt_cfg,
            adapter_ratio=adapter_ratio,
            use_dynamic_gating=use_dynamic_gating,
            ttt_drop_prob=ttt_drop_prob,
        )
        # 3. TCN Head (Classification - 共通)
        self.tcn_head = TCNHead(
            d_features=self.F2,
            n_groups=n_groups,
            tcn_depth=2, 
            kernel_length=4,
            dropout_tcn=0.3,
            n_classes=n_classes
        )

        # ── Reactive (entropy-driven) gating knobs ──────────────────────────────
        self.use_dynamic_gating = use_dynamic_gating
        self.gating_mode = gating_mode
        self.entropy_gating_in_train = entropy_gating_in_train
        if self.use_dynamic_gating and self.gating_mode == "entropy":
            # alpha gate and lr_scale gate are separated for flexibility
            self.entropy_alpha_gate = EntropyGating(
                init_w=entropy_alpha_init_w,
                init_b=entropy_alpha_init_b,
                threshold=entropy_threshold,
                max_out=alpha_max,
                hard_off_below_threshold=True,
            )
            self.entropy_lr_gate = EntropyGating(
                init_w=entropy_lr_init_w,
                init_b=entropy_lr_init_b,
                threshold=entropy_threshold,
                max_out=lr_scale_max,
                hard_off_below_threshold=True,
            )
        else:
            self.entropy_alpha_gate = None
            self.entropy_lr_gate = None

        # Debug buffers for reactive gating
        self.pmax_threshold = pmax_threshold  # Only apply TTT when pmax < this value
        self.last_gate_entropy = None   # [B]
        self.last_ttt_lr_scale = None   # [B]
        
        # 2-pass debug buffers (Pass 1 = attention-only)
        self.last_logits_pass1 = None   # [B, n_classes]
        self.last_probs_pass1 = None    # [B, n_classes]
        self.last_preds_pass1 = None    # [B]
        self.last_entropy_pass1 = None  # [B] (raw, not normalized)
        self.last_pmax_pass1 = None     # [B]
        
        # 2-pass debug buffers (Pass 2 = with TTT)
        self.last_logits_pass2 = None   # [B, n_classes]
        self.last_probs_pass2 = None    # [B, n_classes]
        self.last_preds_pass2 = None    # [B]
        self.last_entropy_pass2 = None  # [B]
        self.last_pmax_pass2 = None     # [B]
        
        # Delta metrics
        self.last_delta_logits = None   # [B]
        self.last_delta_kl = None       # [B]
        
        # Update info
        self.last_alpha = None          # [B]
        self.last_update_on = None      # [B] boolean

    def forward(self, x, cache_params: TTTCache | None = None):
        # Input: [B, C, T] or [B, 1, C, T]
        if x.ndim == 4: x = x.squeeze(1) 
            
        # 1. Conv Feature Extraction
        x = self.conv_block(x)
        
        # 2. Sequence Learning (Hybrid)
        x = x.permute(0, 2, 1) # [B, D, T] -> [B, T, D]

        # ── Strategy A/B: entropy-driven reactive adaptation (2-pass) ──────────
        # By default, run this ONLY in eval/inference.
        # Running 2-pass entropy gating during training doubles compute and adds noise (dropout),
        # often harming stability. Enable explicitly via entropy_gating_in_train=True.
        if self.use_dynamic_gating and self.gating_mode == "entropy" and (self.entropy_gating_in_train or (not self.training)):
            # Pass 1: Attention-only (NO TTT) to get provisional logits -> entropy
            # NOTE: During training with entropy_gating_in_train=True, we run pass1 in eval mode
            # to avoid dropout noise affecting entropy estimation
            was_training = self.training
            if was_training and self.entropy_gating_in_train:
                self.hybrid_encoder.eval()
                self.tcn_head.eval()
            
            x_main = self.hybrid_encoder(x, enable_ttt=False, cache_params=cache_params)
            x_main_t = x_main.permute(0, 2, 1)  # [B,T,D] -> [B,D,T]
            with torch.no_grad():
                logits_main = self.tcn_head(x_main_t)  # [B, n_classes]
                p = torch.softmax(logits_main, dim=-1).clamp_min(1e-12)
                raw_entropy = -(p * p.log()).sum(dim=-1)  # [B]
                # CRITICAL FIX: Normalize entropy to [0, 1] for cross-dataset compatibility
                # max_entropy = ln(K) where K is number of classes
                n_classes = logits_main.shape[-1]
                max_entropy = torch.log(torch.tensor(float(n_classes), device=logits_main.device))
                gate_entropy = raw_entropy / max_entropy  # [B], normalized to [0, 1]
                
                # Store Pass 1 debug info
                self.last_logits_pass1 = logits_main.detach()
                self.last_probs_pass1 = p.detach()
                self.last_preds_pass1 = logits_main.argmax(dim=-1).detach()
                self.last_entropy_pass1 = raw_entropy.detach()
                self.last_pmax_pass1 = p.max(dim=-1).values.detach()  # [B]

            # Restore training mode for pass2
            if was_training and self.entropy_gating_in_train:
                self.hybrid_encoder.train()
                self.tcn_head.train()

            # Compute alpha and lr_scale from normalized entropy
            alpha_1d = self.entropy_alpha_gate(gate_entropy)  # [B]
            lr_scale = self.entropy_lr_gate(gate_entropy)     # [B]

            # Apply pmax filter: only adapt when model is uncertain (pmax < threshold)
            pmax = p.max(dim=-1).values  # [B]
            if self.pmax_threshold < 1.0:
                pmax_mask = (pmax < self.pmax_threshold).to(alpha_1d.dtype)  # [B]
                alpha_1d = alpha_1d * pmax_mask
                lr_scale = lr_scale * pmax_mask

            # Debug buffers
            self.last_gate_entropy = gate_entropy.detach()
            self.last_ttt_lr_scale = lr_scale.detach()
            self.last_alpha = alpha_1d.detach()  # [B]
            self.last_update_on = (alpha_1d > 0).detach()  # [B] boolean mask

            # Pass 2: Run full hybrid with externally supplied alpha and lr_scale
            gate_alpha = alpha_1d.view(-1, 1, 1)
            x = self.hybrid_encoder(x, gate_alpha=gate_alpha, enable_ttt=True, lr_scale=lr_scale, cache_params=cache_params)
        else:
            # Legacy path (feature-stats gating or static scale)
            self.last_gate_entropy = None
            self.last_ttt_lr_scale = None
            self.last_logits_pass1 = None
            self.last_probs_pass1 = None
            self.last_preds_pass1 = None
            self.last_entropy_pass1 = None
            self.last_pmax_pass1 = None
            self.last_alpha = None
            self.last_update_on = None
            x = self.hybrid_encoder(x, cache_params=cache_params)

        x = x.permute(0, 2, 1) # [B, T, D] -> [B, D, T]
        
        # 3. Classification
        logits = self.tcn_head(x)
        
        # ── Compute Pass 2 debug info & delta metrics (for 2-pass analysis) ──
        if self.last_logits_pass1 is not None:
            with torch.no_grad():
                # Pass 2 info
                p2 = torch.softmax(logits, dim=-1).clamp_min(1e-12)
                self.last_logits_pass2 = logits.detach()
                self.last_probs_pass2 = p2.detach()
                self.last_preds_pass2 = logits.argmax(dim=-1).detach()
                self.last_entropy_pass2 = -(p2 * p2.log()).sum(dim=-1).detach()  # [B]
                self.last_pmax_pass2 = p2.max(dim=-1).values.detach()  # [B]
                
                # Delta metrics
                self.last_delta_logits = (logits - self.last_logits_pass1).norm(dim=-1).detach()  # [B]
                # KL(p1 || p2) per sample
                p1 = self.last_probs_pass1.clamp_min(1e-12)
                self.last_delta_kl = (p1 * (p1.log() - p2.log())).sum(dim=-1).detach()  # [B]
        else:
            self.last_logits_pass2 = None
            self.last_probs_pass2 = None
            self.last_preds_pass2 = None
            self.last_entropy_pass2 = None
            self.last_pmax_pass2 = None
            self.last_delta_logits = None
            self.last_delta_kl = None
        
        return logits

    def get_debug_batch(self):
        """
        Return latest per-batch debug tensors (if available).
        This is used by ClassificationModule.test_step to save small json stats.
        """
        dbg = {
            "alpha": getattr(self, "last_alpha", None),  # Use self.last_alpha from forward
            "gate_entropy": getattr(self, "last_gate_entropy", None),
            "ttt_lr_scale": getattr(self, "last_ttt_lr_scale", None),
            "clip_ratio": None,
            "group_attn_weights": getattr(self.conv_block, "last_group_attn_weights", None),
            "conv_norm_pre": getattr(self.conv_block, "last_conv_norm_pre", None),
            "conv_norm_post": getattr(self.conv_block, "last_conv_norm_post", None),
            "group_attn_gamma": getattr(self.conv_block, "last_group_attn_gamma", None),
            "group_attn_temperature": getattr(getattr(self.conv_block, "group_attn", None), "last_temperature", None),
            
            # ── 2-pass debug info ──
            # Pass 1 (attention-only)
            "logits_pass1": getattr(self, "last_logits_pass1", None),
            "probs_pass1": getattr(self, "last_probs_pass1", None),
            "preds_pass1": getattr(self, "last_preds_pass1", None),
            "entropy_pass1": getattr(self, "last_entropy_pass1", None),
            "pmax_pass1": getattr(self, "last_pmax_pass1", None),
            
            # Pass 2 (with TTT)
            "logits_pass2": getattr(self, "last_logits_pass2", None),
            "probs_pass2": getattr(self, "last_probs_pass2", None),
            "preds_pass2": getattr(self, "last_preds_pass2", None),
            "entropy_pass2": getattr(self, "last_entropy_pass2", None),
            "pmax_pass2": getattr(self, "last_pmax_pass2", None),
            
            # Delta metrics
            "delta_logits": getattr(self, "last_delta_logits", None),
            "delta_kl": getattr(self, "last_delta_kl", None),
            
            # Update info
            "update_on": getattr(self, "last_update_on", None),  # boolean [B]
        }

        # Clip ratio: average across layers (if available)
        ratios = []
        eff_lrs = []
        for layer in getattr(self.hybrid_encoder, "layers", []):
            try:
                ttt_layer = layer.ttt_adapter.ttt_layer
                r = getattr(ttt_layer, "last_clip_ratio_per_sample", None)
                if r is not None:
                    ratios.append(r)
                elr = getattr(ttt_layer, "last_effective_base_lr_per_sample", None)
                if elr is not None:
                    eff_lrs.append(elr)
            except Exception:
                continue
        if ratios:
            dbg["clip_ratio"] = torch.stack(ratios, dim=0).mean(dim=0).detach()
        if eff_lrs:
            dbg["ttt_effective_base_lr"] = torch.stack(eff_lrs, dim=0).mean(dim=0).detach()
        return dbg

    def create_cache(self, batch_size: int) -> TTTCache:
        """Create a new TTTCache for stateful online inference."""
        return TTTCache(model=self, batch_size=batch_size)

    def get_num_ttt_layers(self) -> int:
        """Return the number of TTT layers (same as trans_depth)."""
        return self.hybrid_encoder.trans_depth

class TCFormerHybrid(ClassificationModule):
    def __init__(self, n_classes, **kwargs):
        # Extract TTT specific args safely
        ttt_config = kwargs.get("ttt_config", {})
        
        model = TCFormerHybridModule(
            n_classes=n_classes,
            Chans=kwargs.get("n_channels", 22),
            ttt_config=ttt_config,
            # Pass other params from kwargs or defaults
            trans_depth=kwargs.get("trans_depth", 2),
            q_heads=kwargs.get("q_heads", 4),
            kv_heads=kwargs.get("kv_heads", 4),
            adapter_ratio=kwargs.get("adapter_ratio", 0.25),
            use_group_attn=kwargs.get("use_group_attn", True),
            use_dynamic_gating=kwargs.get("use_dynamic_gating", False),
            gating_mode=kwargs.get("gating_mode", "feature_stats"),
            entropy_alpha_init_w=kwargs.get("entropy_alpha_init_w", 2.0),
            entropy_alpha_init_b=kwargs.get("entropy_alpha_init_b", -3.0),
            entropy_lr_init_w=kwargs.get("entropy_lr_init_w", 2.0),
            entropy_lr_init_b=kwargs.get("entropy_lr_init_b", -2.0),
            entropy_threshold=kwargs.get("entropy_threshold", 0.95),
            alpha_max=kwargs.get("alpha_max", 0.5),
            lr_scale_max=kwargs.get("lr_scale_max", 0.5),
            entropy_gating_in_train=kwargs.get("entropy_gating_in_train", False),
            ttt_drop_prob=kwargs.get("ttt_drop_prob", 0.0),
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)

