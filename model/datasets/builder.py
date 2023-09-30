"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""


from model.utils.registry import Registry

DATASETS = Registry('datasets')


def build_dataset(cfg):
     
    return DATASETS.build(cfg)
