import argparse
import munch
import yaml
import os


def get_args(string_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help='path to config file')
    parser.add_argument('--model_cfg', default="implicit_mlp.yaml", type=str, help='path to config file')
    parser.add_argument('--dir', default="debug", type=str, help='save directory')
    parser.add_argument('--epochs', default="100", type=int, help='training epochs')
    parser.add_argument('--device', default="cuda", type=str, help='choose cuda or cpu')
    parser.add_argument('--fp16', default="True", type=bool, help='use gradscaling')
    parser.add_argument('--seed', default="0", type=int, help='seed')
    parser.add_argument('--parameter_list', default="configs/data.yaml", type=str, help='parameter_specification')
    parser.add_argument('--wildcard', type=int, help='do anything with this argument')

    #'experiment args'
    parser.add_argument('--encoding', default="none", type=str, help='parameter_specification')
    if string_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(string_args)

    args.config = os.path.join("configs", args.config)
    args.model_cfg = os.path.join("configs/model_cfg/", args.model_cfg)
    base = "experiments"
    args.dir_name = args.dir
    args.dir = os.path.join(base, args.dir)
    return args


def get_config(config_path):
    cfg_txt = open(config_path, 'r').read()
    cfg = munch.Munch.fromDict(yaml.safe_load(cfg_txt))
    return cfg
