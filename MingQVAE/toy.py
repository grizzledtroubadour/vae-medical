import argparse, os, sys, datetime
from omegaconf import OmegaConf

import numpy as np
from pytorch_lightning.trainer import Trainer
import pytorch_lightning as pl

#from pl_module import get_checkpoint_callback, get_logger, get_callbacks
#from tools.imp import instantiate_from_config

# 命令行指令
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-p", "--project", type=str, help="name of new or path to existing project")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml", default=list(), help="paths to base configs. Loaded from left-to-right.\nParameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("-t", "--train", action='store_true', help="train or not")
    parser.add_argument("--no-test", action='store_true', help="disable test")
    parser.add_argument("-d", "--debug", action='store_true', help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, help="post-postfix for default name")
    return parser

if __name__ == "__main__":
    now_time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") # 记录开始时间
    sys.path.append(os.getcwd()) # 将本脚本所在文件夹添加到环境路径

    parser = get_parser()
    #parser = Trainer.add_argparse_args(parser) # 为Namespace加入了很多Trainer相关的变量
    opt, unknown = parser.parse_known_args()

    print(opt)
    print(unknown)

    cli = OmegaConf.from_dotlist(unknown) # unknown是命令行未识别命令
    print(cli)

    a = {'data': np.array([[3,2,1,0,-1,-2],[1,2,3,4,5,6]])}
    print(a)
    b = a["data"].reshape(-1,2,3)
    print(b)
    print(b[0,0])
