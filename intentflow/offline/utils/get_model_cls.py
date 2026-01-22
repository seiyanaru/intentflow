def get_model_cls(model_name):
    if model_name.lower() == "tcformer":
        from models.tcformer.tcformer import TCFormer
        return TCFormer
    elif model_name.lower() == "tcformer_ttt":
        from models.tcformer_ttt.tcformer_ttt import TCFormerTTT
        return TCFormerTTT
    elif model_name.lower() == "tcformer_hybrid":
        from models.tcformer_ttt.tcformer_hybrid import TCFormerHybrid
        return TCFormerHybrid
    elif model_name.lower() == "tcformer_otta":
        from models.tcformer_otta import TCFormerOTTA
        return TCFormerOTTA
    elif model_name == "ATCNet":
        from models.tcformer.atcnet import ATCNet
        return ATCNet
    elif model_name == "BaseNet":
        from models.tcformer.basenet import BaseNet
        return BaseNet
    # Add other models as needed, avoiding top-level imports that cause errors
    # if dependencies are missing.
    
    # Default behavior (might fail if dependencies are missing)
    # from models import ...
    
    raise NotImplementedError(f"Model {model_name} not properly configured in get_model_cls or missing dependencies.")

model_dict = {} # Dummy for compatibility if imported elsewhere
