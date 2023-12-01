import os
import argparse
import collections

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from model.models import build_model
from model.datasets import build_dataset
from model.datasets.utils import collate_fn
from model.utils.config import Config, DictAction
from model.utils.logger import get_root_logger
from model.utils.env import get_random_seed, set_seed
from model.engines.test import TEST


def get_parser():
    parser = argparse.ArgumentParser(description='Pointcept Test Process')
    parser.add_argument('--config', default="configs/semseg-scannet.py",
                        metavar="FILE", help="path to config file")
    parser.add_argument('--options', nargs='+',
                        action=DictAction, help='custom options')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()

    # config_parser
    cfg = Config.fromfile(args.config)

    if args.options is not None:
        cfg.merge_from_dict(args.options)

    torch.cuda.set_device(cfg.test_set_gpu)

    if cfg.seed is None:
        cfg.seed = get_random_seed()
    torch.cuda.set_device(cfg.test_set_gpu)

    os.makedirs(cfg.save_path, exist_ok=True)

    # default_setup
    set_seed(cfg.seed)
    # TODO: add support to multi gpu test
    cfg.batch_size_val_per_gpu = cfg.batch_size_test
    cfg.num_worker_per_gpu = cfg.num_worker  # TODO: add support to multi gpu test

    # tester init
    weight_name = os.path.basename(cfg.weight).split(".")[0]
    logger = get_root_logger(log_file=os.path.join(
        cfg.save_path, "test-{}.log".format(weight_name)))
    logger.info("=> Loading config ...")
    logger.info(f"Save path: {cfg.save_path}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # build model
    logger.info("=> Building model ...")
    model = build_model(cfg.model).cuda()
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")

    # build dataset
    logger.info("=> Building test dataset & dataloader ...")
    test_dataset = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size_val_per_gpu,
                                              shuffle=False,
                                              num_workers=cfg.num_worker_per_gpu,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    # load checkpoint
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(
            cfg.weight, map_location=torch.device('cuda', cfg.test_set_gpu))
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # module.xxx.xxx -> xxx.xxx
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        checkpoint['state_dict'] = model.state_dict()
        logger.info("=> loaded weight '{}' (epoch {})".format(
            cfg.weight, checkpoint['epoch']))
        cfg.epochs = checkpoint['epoch']  # TODO: move to self
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.weight))
    TEST.build(cfg.test)(cfg, test_loader, model)


if __name__ == '__main__':
    main()
