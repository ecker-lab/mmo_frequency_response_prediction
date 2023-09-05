import numpy as np
import torch
import wandb
import time
import os
from acousticnn.mmo import get_listdataloader, Iter_Dataset, get_dataloader
from acousticnn.mmo import train
from acousticnn.utils.builder import build_opti_sche, build_model
from acousticnn.utils.logger import init_train_logger, print_log
from acousticnn.utils.argparser import get_args, get_config
from torchinfo import summary


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def main():
    args = get_args()
    config = get_config(args.config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.model.input_encoding = "none"

    start_wandb(args, config)
    logger = init_train_logger(args, config)
    n_samples = 1000
    parameter_list = get_config(args.parameter_list)
    trainloader = get_listdataloader(args, config, parameter_list, n_samples, batch_size=config.batch_size)
    parameters = parameter_list["sample1"]
    parameters["n_masses"] = 5
    valloader = get_dataloader(args, config, test=True, n_samples=1000, parameters=parameters, batch_size=100)
    net = build_model(trainloader, args, config)
    optimizer, scheduler = build_opti_sche(net, config)
    train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)


def start_wandb(args, config):
    wandb.init(project=f"{config.model.name}", entity="jans")
    #wandb.config.update(args)
    wandb.config.update(config)
    wandb.run.name = args.dir_name + str(time.time())


if __name__ == '__main__':
    main()
