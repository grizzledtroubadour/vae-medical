import argparse, os, sys, datetime, glob
from numpy.core.shape_base import vstack
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from tools.imp import instantiate_from_config


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="check the points in the latent")
    parser.add_argument("-r", "--resume", type=str, required=True, help="resume from logdir or checkpoint in logdir")
    opt = parser.parse_args()
    
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        paths = opt.resume.split("/")
        try:
            idx = len(paths)-paths[::-1].index("logs")+1
        except ValueError:
            idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
        logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), opt.resume
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
    print(f"logdir:{logdir}")
    base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))

    configs = [OmegaConf.load(cfg) for cfg in base_configs]
    config = OmegaConf.merge(*configs)

    model = instantiate_from_config(config.model).cuda()
    encoder = model.encoder.eval()

    dataset = instantiate_from_config(config.data.params.train)

    dataloader = DataLoader(dataset, batch_size=48, num_workers=24)

    out = np.empty((0, config.model.params.ddconfig.z_channels))
    for i, data in tqdm(enumerate(dataloader)):

        imgs = data['image'].cuda()
        
        latents = encoder(imgs.permute(0,3,1,2)).permute(0,2,3,1)
        latents = latents.reshape(-1, latents.shape[3]).cpu().detach().numpy()

        out = vstack((out, latents))

        
    cachedir = 'asserts/latent_space'
    os.makedirs(cachedir, exist_ok=True)
        
    np.save(os.path.join(cachedir, "latent_vec.npy"), out)

    embeddings = model.quantize.embedding.weight.cpu().detach().numpy()
    np.save(os.path.join(cachedir, "latent_emb.npy"), embeddings)

    