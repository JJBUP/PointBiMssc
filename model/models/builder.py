from model.utils.registry import Registry

MODELS = Registry('models')
MODULES = Registry('modules')


def build_model(cfg):
     
    return MODELS.build(cfg)
