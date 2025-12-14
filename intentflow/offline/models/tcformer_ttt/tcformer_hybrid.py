import torch
import torch.nn as nn
from einops import rearrange
from models.tcformer.tcformer import MultiKernelConvBlock, TCNHead, ClassificationModule
from models.tcformer.tcformer import _GQAttention, DropPath, _build_rotary_cache 
from models.tcformer_ttt.ttt_layer import TTTConfig, TTTLinear

class TTTAdapter(nn.Module):
    """
    TTTを用いた軽量アダプターモジュール
    Structure: DownProj -> TTT-Linear -> UpProj
    ボトルネック構造により計算コストを削減しつつ適応を行う。
    """
    def __init__(self, config: TTTConfig, input_dim: int, adapter_dim: int):
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
        self.ttt_layer = TTTLinear(adapter_config, layer_idx=0)
        self.norm = nn.LayerNorm(adapter_dim)
        
        # Zero-Init: 学習初期はアダプターの影響をゼロにする（安定性確保のため）
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # x: [B, T, D]
        residual = x
        
        # 1. 圧縮 (Down-projection)
        x = self.down_proj(x)
        x = self.act(x)
        
        # 2. 適応 (TTT: Test-Time Training)
        # ここで入力データに応じた重み更新が行われる
        x = self.ttt_layer(x) 
        
        # 3. 復元 (Up-projection)
        x = self.norm(x)
        x = self.up_proj(x)
        
        return x

class HybridBlock(nn.Module):
    """
    Attention (Fixed) + TTT Adapter (Adaptive) の並列ハイブリッドブロック
    """
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, 
                 ttt_config: TTTConfig, adapter_ratio=0.25, dropout=0.4, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        
        # --- Main Path: Fixed Self-Attention (普遍的な特徴抽出) ---
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        
        # --- Adapter Path: TTT (個人差への適応) ---
        # adapter_ratio (例: 0.25) に圧縮して計算コストを下げる
        adapter_dim = int(d_model * adapter_ratio)
        self.ttt_adapter = TTTAdapter(ttt_config, d_model, adapter_dim)
        
        # Gating parameter (学習可能な係数 alpha)
        # 初期値0.0 -> 学習が進むにつれてTTTを信頼するようになる
        self.adapter_scale = nn.Parameter(torch.tensor(0.0))
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP Part (Shared)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        # Pre-Norm
        normed_x = self.norm1(x)
        
        # Parallel Execution (並列実行)
        
        # 1. Main Path: Attention (Context)
        # 既存の重みで安定した特徴を抽出
        out_attn = self.attn(normed_x, cos, sin)
        
        # 2. Adapter Path: TTT (Adaptation)
        # その場のデータで適応した補正値を計算
        out_ttt = self.ttt_adapter(normed_x)
        
        # Integration (統合)
        # X_new = X + Attention(X) + alpha * TTT(X)
        # alpha (adapter_scale) がTTTの影響度を制御する
        x = x + self.drop_path(out_attn + self.adapter_scale * out_ttt)
        
        # MLP Block
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class HybridEncoder(nn.Module):
    def __init__(self, d_model, trans_depth, q_heads, kv_heads, ttt_config, adapter_ratio=0.25, drop_path_max=0.25):
        super().__init__()
        self.d_model = d_model
        
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
                drop_path_rate=dpr[i]
            )
            for i in range(trans_depth)
        ])

    def _rotary_cache(self, seq_len: int, device: torch.device):
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(self.head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

    def forward(self, x):
        # x: [B, T, D]
        seq_len = x.shape[1]
        cos, sin = self._rotary_cache(seq_len, x.device)
        
        for layer in self.layers:
            x = layer(x, cos[:seq_len], sin[:seq_len])
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
            use_group_attn=False 
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
            ttt_reg_lambda=ttt_config.get("ttt_reg_lambda", 0.01) # 正則化
        )
        
        self.hybrid_encoder = HybridEncoder(
            d_model=self.F2,
            trans_depth=trans_depth,
            q_heads=q_heads,
            kv_heads=kv_heads,
            ttt_config=self.ttt_cfg,
            adapter_ratio=adapter_ratio
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

    def forward(self, x):
        # Input: [B, C, T] or [B, 1, C, T]
        if x.ndim == 4: x = x.squeeze(1) 
            
        # 1. Conv Feature Extraction
        x = self.conv_block(x)
        
        # 2. Sequence Learning (Hybrid)
        x = x.permute(0, 2, 1) # [B, D, T] -> [B, T, D]
        x = self.hybrid_encoder(x) 
        x = x.permute(0, 2, 1) # [B, T, D] -> [B, D, T]
        
        # 3. Classification
        x = self.tcn_head(x)
        return x

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
            adapter_ratio=kwargs.get("adapter_ratio", 0.25)
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)

