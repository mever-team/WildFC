from .data_mfm import build_loader_mfm
from .data_finetune import build_loader_finetune, build_loader_test

def build_loader(config, logger, is_pretrain, is_test):
    if is_pretrain:
        return build_loader_mfm(config, logger)
    elif is_test:
        return build_loader_test(config, logger)
    else:
        return build_loader_finetune(config, logger)
