import os

from model.engines.defaults import default_argument_parser, default_config_parser, default_setup
from model.engines.train import Trainer
from model.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = Trainer(cfg)
    trainer.train()


def main():
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_gpu = [1, 2, 3]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
