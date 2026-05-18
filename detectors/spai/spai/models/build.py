from .swin_transformer import build_swin
from .vision_transformer import build_vit
from .sid import build_cls_vit, build_mf_vit
from .mfm import build_mfm


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_mfm(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        elif model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model


def build_cls_model(config):
    model_type = config.MODEL.TYPE
    task_type = config.MODEL.SID_APPROACH
    if model_type == "vit" and task_type == "single_extraction":
        model = build_cls_vit(config)
    elif model_type == "vit" and task_type == "freq_restoration":
        model = build_mf_vit(config)
    else:
        raise NotImplementedError(f"Unknown cls model: {model_type}")
    return model
