import torch
import torch.nn as nn
from einops import rearrange
from models.tcformer.tcformer import MultiKernelConvBlock, TCNHead, ClassificationHead, ClassificationModule
from models.tcformer_ttt.ttt_layer import TTTConfig, TTTBlock, TTTCache

class TTTEncoder(nn.Module):
    def __init__(self, ttt_config: TTTConfig):
        super().__init__()
        self.config = ttt_config
        self.layers = nn.ModuleList(
            [TTTBlock(ttt_config, layer_idx=i) for i in range(ttt_config.num_hidden_layers)]
        )

    def forward(self, x):
        # x: [B, T, D]
        for layer in self.layers:
            # TTTBlock expects (B, T, D)
            x = layer(x)
        return x

class TCFormerTTT(ClassificationModule):
    def __init__(
        self,
        n_classes,
        Chans=22, # Default as it might be passed as n_channels or Chans depending on caller
        n_channels=22, # Compatibility alias
        F1=32,
        D=2,
        d_group=16,
        temp_kernel_lengths=[20, 32, 64],
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        dropout_clf=0.5,
        ttt_config=None,
        **kwargs # Catch all other arguments like use_group_attn, q_heads, etc.
    ):
        
        # Resolve n_channels
        if 'n_channels' in kwargs:
             Chans = kwargs['n_channels']
        
        # We need to initialize ClassificationModule with 'self' as the model, but 'self' is not fully initialized yet.
        # ClassificationModule expects a 'model' argument in its __init__ and then calls self.model = model.
        # It is a wrapper class that inherits from LightningModule.
        
        # Usually usage is:
        # class MyModel(ClassificationModule):
        #    def __init__(self, ...):
        #        model_backbone = ...
        #        super().__init__(model=model_backbone, n_classes=n_classes, **kwargs)
        
        # So we should define the backbone separately or use self as the model if we construct it first?
        # TCFormer implementation:
        # class TCFormer(ClassificationModule):
        #     def __init__(self, ...):
        #         model = TCFormerModule(...)
        #         super().__init__(model, n_classes, **kwargs)
        
        # So we need a TCFormerTTTModule (backbone) and TCFormerTTT (wrapper).
        
        # Let's refactor:
        # 1. Rename current TCFormerTTT class to TCFormerTTTModule and make it inherit nn.Module
        # 2. Create new TCFormerTTT class inheriting ClassificationModule that wraps TCFormerTTTModule
        
        # But I am editing in place. I will rename the current class signature first to TCFormerTTTModule
        # And then add the wrapper class at the end.
        
        pass # Placeholder for now, logic below will replace the class definition
        
# Re-implementation with split classes

class TCFormerTTTModule(nn.Module):
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
        **kwargs  # Added to accept hybrid_config and other arguments
    ):
        super().__init__()
        
        # 1. MultiKernelConvBlock
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
        
        # 2. TTT Encoder
        if ttt_config is None:
            ttt_config = {}
            
        # Extract Hybrid config if present
        hybrid_config = kwargs.get("hybrid_config", {})
        
        # Determine effective TTT parameters (override with hybrid_config if present)
        base_lr = hybrid_config.get("lr", ttt_config.get("base_lr", 1.0))
        reg_lambda = hybrid_config.get("reg", ttt_config.get("ttt_reg_lambda", 0.0))
        ratio = hybrid_config.get("ratio", 1.0)

        self.ttt_cfg = TTTConfig(
            hidden_size=self.F2, 
            num_hidden_layers=ttt_config.get("trans_depth", 2),
            num_attention_heads=ttt_config.get("q_heads", 4),
            hidden_act="silu",
            ttt_layer_type=ttt_config.get("layer_type", "linear"),
            ttt_base_lr=base_lr,
            mini_batch_size=ttt_config.get("mini_batch_size", 16),
            share_qk=ttt_config.get("share_qk", False),
            use_dual_form=ttt_config.get("use_dual_form", True),
            learnable_init_state=ttt_config.get("learnable_init_state", False),
            conv_kernel=4,
            intermediate_size=self.F2 * 4,
            # Pass Hybrid parameters
            hybrid_ratio=ratio, 
            ttt_reg_lambda=reg_lambda,
            hybrid_reg=reg_lambda, # Keep for consistency if accessed elsewhere
        )
        
        self.ttt_encoder = TTTEncoder(self.ttt_cfg)

        # 3. TCN Head
        self.tcn_head = TCNHead(
            d_features=self.F2,
            n_groups=n_groups,
            tcn_depth=2, 
            kernel_length=4,
            dropout_tcn=0.3,
            n_classes=n_classes
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1) 
            
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)
        x = self.ttt_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.tcn_head(x)
        return x

class TCFormerTTT(ClassificationModule):
    def __init__(
        self,
        n_classes,
        Chans=22,
        n_channels=22,
        F1=32,
        D=2,
        d_group=16,
        temp_kernel_lengths=[20, 32, 64],
        pool_length_1=8,
        pool_length_2=7,
        dropout_conv=0.4,
        dropout_clf=0.5,
        ttt_config=None,
        **kwargs 
    ):
        if 'n_channels' in kwargs:
             Chans = kwargs['n_channels']
        elif n_channels is not None:
             Chans = n_channels

        model = TCFormerTTTModule(
            n_classes=n_classes,
            Chans=Chans,
            F1=F1,
            D=D,
            d_group=d_group,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            dropout_conv=dropout_conv,
            dropout_clf=dropout_clf,
            ttt_config=ttt_config
        )
        super().__init__(model=model, n_classes=n_classes, **kwargs)


