import numpy as np
from acousticnn.mmo import get_dataloader, Iter_Dataset
from acousticnn.mmo import train
from acousticnn.utils.builder import build_opti_sche, build_model
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from torchinfo import summary
import wandb 
import time
import os
import torch

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()

    config = get_config(args.config)
    logger = init_train_logger(args, config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    parameters = {"n_masses": 4,
                  "sample_f": False,
                  "f_per_sample": 200,
                  "sample_m": True,
                  "m_range": (1, 25),
                  "sample_d": True,
                  "d_range": (0.1, 1),
                  "sample_k": True,
                  "k_range": (1, 30),
                  "normalize": True,
                  "normalize_factor": 10
                  }

    if args.encoding == "none":
        parameters["normalize"] = True
        parameters["normalize_factor"] = 10
        config.model.input_encoding = "none"
    elif args.encoding == "random":
        parameters["normalize"] = True
        parameters["normalize_factor"] = 10
        config.model.input_encoding = "random"
    elif args.encoding == "sin":
        parameters["normalize"] = True
        parameters["normalize_factor"] = 100
        config.model.input_encoding = "sin"
    else:
        raise NotImplementedError
    start_wandb(args, config)

    trainloader = get_dataloader(args, config, test=False, n_samples=4000, parameters=parameters, batch_size=config.batch_size)
    valloader = get_dataloader(args, config, test=True, n_samples=1000, parameters=parameters, batch_size=100)

    net = build_model(trainloader, args, config)
    a, parameters = next(iter(trainloader))[:2]
    print_log(summary(net, input_data=(a.to(args.device), [par.to(args.device) for par in parameters])), logger=logger)
    optimizer, scheduler = build_opti_sche(net, config)
    train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)


def start_wandb(args, config):
    wandb.init()
    wandb.config.update(config)
    wandb.run.name = args.dir_name + str(time.time())


if __name__ == '__main__':
    main()
