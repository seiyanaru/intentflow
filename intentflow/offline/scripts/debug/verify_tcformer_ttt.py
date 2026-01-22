import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath("/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline"))

from utils.get_model_cls import get_model_cls
# from configs.tcformer_ttt.tcformer_ttt import ttt_config_dict # Won't work directly, need to load yaml

import yaml
from models.tcformer_ttt.tcformer_ttt import TCFormerTTT

def verify_tcformer_ttt():
    print(">>> Verifying TCFormerTTT implementation...")
    
    # 1. Load config
    config_path = "/workspace-cloud/seiya.narukawa/intentflow/intentflow/offline/configs/tcformer_ttt/tcformer_ttt.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Extract model args
    model_kwargs = config['model_kwargs']
    ttt_config = model_kwargs.pop('ttt_config') # Separate ttt config
    
    # 3. Instantiate model
    print(">>> Instantiating model...")
    try:
        # Merge model_kwargs into kwargs explicitly if needed, or pass individual args
        # But TCFormerTTT takes **kwargs logic somewhat if we adapt it or pass explicit args
        # Let's check TCFormerTTT init again. It takes explicit args.
        # We need to map model_kwargs to explicit args.
        
        F1 = model_kwargs.get('F1', 32)
        D = model_kwargs.get('D', 2)
        d_group = model_kwargs.get('d_group', 16)
        temp_kernel_lengths = model_kwargs.get('temp_kernel_lengths', [20, 32, 64])
        pool_length_1 = model_kwargs.get('pool_length_1', 8)
        pool_length_2 = model_kwargs.get('pool_length_2', 7)
        dropout_conv = model_kwargs.get('dropout_conv', 0.4)
        dropout_clf = 0.5 # Default or from config if available (it is not in model_kwargs in yaml usually but let's assume default)
        
        model = TCFormerTTT(
            n_classes=4,
            Chans=22,
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
        print(">>> Model instantiated successfully.")
    except Exception as e:
        print("!!! Error instantiating model: {}".format(e))
        raise e
        
    # 4. Create dummy data
    batch_size = 2
    channels = 22
    time_steps = 1000 # 4s * 250Hz
    
    x = torch.randn(batch_size, channels, time_steps)
    print(">>> Dummy data shape: {}".format(x.shape))
    
    # 5. Forward pass
    print(">>> Running forward pass...")
    try:
        y = model(x)
        print(">>> Output shape: {}".format(y.shape))
        
        expected_shape = (batch_size, 4)
        if y.shape == expected_shape:
            print(">>> Forward pass successful! Output shape matches expected.")
        else:
            print("!!! Forward pass failed! Expected {}, got {}".format(expected_shape, y.shape))
            
    except Exception as e:
        print("!!! Error during forward pass: {}".format(e))
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    verify_tcformer_ttt()

