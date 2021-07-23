import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf

import pytorch_lightning as pl

from pl_module import get_checkpoint_callback, get_logger, get_callbacks
from tools.imp import instantiate_from_config

# 命令行指令
def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, # 出现-n但无参数时，默认为True
        default="", nargs="?", # 需要一个或不需要参数
        help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, nargs="?", const=True, default="", 
        help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-p", "--project", type=str, help="name of new or path to existing project")
    parser.add_argument("-b", "--base", nargs="*", # 可接多个参数
        metavar="base_config.yaml", default=list(), help="paths to base configs. Loaded from left-to-right.\nParameters can be overwritten or added with command-line options of the form `--key value`.")
    parser.add_argument("-t", "--train", action='store_true', help="train or not")
    parser.add_argument("--no-test", action='store_true', help="disable test")
    parser.add_argument("-d", "--debug", action='store_true', help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    return parser

# 找出所有Trainer指定命令与opt中不同配置的变量名
def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser() 
    parser = pl.Trainer.add_argparse_args(parser) # 单独获取Trainer所需的Namespace
    args = parser.parse_args([])
    return sorted(k for k in vars(args).keys() if getattr(opt, k) != getattr(args, k)) # vars返回对象object属性和属性值的字典对象

if __name__ == "__main__":
    now_time = datetime.datetime.now().strftime("%Y%m%d-T%H-%M") # 记录开始时间
    sys.path.append(os.getcwd()) # 将本脚本所在文件夹添加到环境路径

    parser = get_parser()
    parser = pl.Trainer.add_argparse_args(parser) # 为Namespace加入了很多Trainer相关的变量
    opt, unknown = parser.parse_known_args()

    # -------------------
    #  root path configs
    # -------------------
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.resume:
        assert os.path.exists(opt.resume), f"Cannot find {opt.resume}"
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), f"cannot process resume: {opt.resume}"
            logdir = opt.resume.rstrip("/") # 删除字符串末尾的/
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1] # logs/nowname(logdir_path)/configs
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base: # 如果没有指定模型log保存名，使用配置文件名作为保存名
            cfg_fname = os.path.split(opt.base[0])[-1] # 配置文件名（含后缀）
            cfg_name = os.path.splitext(cfg_fname)[0] # 配置文件名（不含后缀）
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now_time+name+opt.postfix # 模型log名：时间+name+额外信息
        logdir = os.path.join("logs", nowname) # log信息保存目录
        
    # log信息存放位置
    ckpt_dir = os.path.join(logdir, 'checkpoints') # checkpoints保存路径
    cfg_dir = os.path.join(logdir, 'configs') # config信息保存路径
    pl.seed_everything(opt.seed)


    try:
        # -----------------
        #  Trainer configs
        # -----------------
        # 读取配置文件并与命令行配置合并
        configs = [OmegaConf.load(cfg) for cfg in opt.base] # opt.base是指定配置文件列表
        cli = OmegaConf.from_dotlist(unknown) # unknown是命令行未识别命令，格式要求"a.b=3"->yaml "a: b:3"

        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create()) # 提出pytorch_lightning的配置单独使用

        # 更改部分trainer配置
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # distributed backend --> default to ddp
        trainer_config["distributed_backend"] = "ddp"
        # 找出opt中与Trainer默认指令值不同的选项放入trainer配置中
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)

        # 环境配置
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config) # 使用trainer配置创建argparse对象
        lightning_config.trainer = trainer_config
        
        # ---------------
        #  model loading
        # ---------------
        model = instantiate_from_config(config.model)

        # -----------------------
        #  trainer and callbacks
        # -----------------------
        trainer_kwargs = dict()

        # add logger module
        logger_config = lightning_config.get('logger', OmegaConf.create()) # 选择一个不为None的值赋给logger_config
        trainer_kwargs["logger"] = get_logger(logdir, logger_config) # 默认使用TestTubeLogger

        # add checkpoint module
        ckpt_config = lightning_config.get('modelcheckpoint', OmegaConf.create())
        model_monitor = getattr(model, 'monitor') if hasattr(model, 'monitor') else None
        trainer_kwargs['checkpoint_callback'] = get_checkpoint_callback(ckpt_dir, ckpt_config, model_monitor)

        # add callback which sets up log directory
        callbacks_config = lightning_config.get('callbacks', OmegaConf.create())
        callbacks_info = {
                'resume': opt.resume,
                'now': now_time,
                'logdir': logdir,
                'ckptdir': ckpt_dir,
                'cfgdir': cfg_dir,
                'config': config,
                'lightning_config': lightning_config,
            }
        trainer_kwargs["callbacks"] = get_callbacks(callbacks_info, callbacks_config)

        trainer = pl.Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)


        # -------------
        #  data config
        # -------------
        data = instantiate_from_config(config.data) # data.data_module.DataModuleFromConfig (LightningDataModule)
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)

    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
