import os, tarfile, glob, shutil
from zipfile import ZipFile
import yaml
import numpy as np
from tqdm import tqdm
import nibabel as nib
from omegaconf import OmegaConf
from torch.utils.data.dataset import Dataset

from tools.utils import download, retrieve
import data.utils as bdu


class MOODBrainBase(Dataset):
    def __init__(self, config=None):
        self.config = config or dict()
        if not type(self.config)==dict:
            self.config = OmegaConf.to_container(self.config)
        self._prepare()

    def __len__(self):
        return len(self.relpaths)

    def __getitem__(self, i):
        data = nib.load(self.datapaths[i])
        imgs = np.asarray(data.get_fdata())
        #vol = resize(vol, (160, 160, 256))
        imgs = imgs.transpose((2,1,0)) # (z,y,x)->(x,y,z)
        imgs = imgs[:, ::-1, :] #::-1倒序输出

        if self.pxlabelpaths:
            pxlabeldata = nib.load(self.pxlabelpaths[i])
            pxlabels = np.asarray(pxlabeldata.get_fdata())
            #vol = resize(vol, (160, 160, 256))
            pxlabels = pxlabels.transpose((2,1,0)) # (z,y,x)->(x,y,z)
            pxlabels = pxlabels[:, ::-1, :] #::-1倒序输出
        else:
            pxlabels = None

        return imgs, pxlabels

    def _prepare(self):
        raise NotImplementedError()

    def load(self):
        with open(self.txt_filelist, "r") as f:
            lines = f.read().splitlines()
        self.relpaths = [l.split(':') for l in lines]
        self.datapaths = [os.path.join(self.datadir, p[0]) for p in self.relpaths] # main images
        if len(self.relpaths[0]) > 1:
            self.pxlabelpaths = [os.path.join(self.datadir, p.split(';')[1]) for p in self.relpaths] # pixel labels
            self.splabelpaths = [os.path.join(self.datadir, p.split(';')[2]) for p in self.relpaths] # sample labels
        else:
            self.pxlabelpaths = None
            self.splabelpaths = None



class MOODBrainTrain(MOODBrainBase):
    NAME = "MOODBrain_train"
    URL = "http://medicalood.dkfz.de/QtVaXgs8il/brain_train.zip"
    FILES = [
        "brain_train.zip",
    ]
    SIZES = [
        22999718114,
    ]

    def _prepare(self):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")

        if not bdu.is_prepared(self.root):
            # prepare
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    download(self.URL, path)

                print("Extracting {} to {}".format(path, datadir))
                bdu.unpack(path, datadir)

            # list
            filelist = glob.glob(os.path.join(datadir, "brain_train", "*.nii.gz"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist] #绝对路径转为相对路径
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)


class MOODBrainToy(MOODBrainBase):
    NAME = "MOODBrain_toy"
    URL = "http://medicalood.dkfz.de/QtVaXgs8il/brain_toy.zip"
    FILES = [
        "brain_toy.zip",
    ]
    SIZES = [
        134917010,
    ]

    def _prepare(self):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")

        if not bdu.is_prepared(self.root):
            # prepare
            print("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    download(self.URL, path)

                print("Extracting {} to {}".format(path, datadir))
                bdu.unpack(path, datadir)

            filelist = glob.glob(os.path.join(datadir, "toy", "*.nii.gz"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist] #绝对路径转为相对路径
            filelist = sorted(filelist)
            filelist = [p+';'+
                        os.path.join('./toy_label/pixel', os.path.basename(p))+';'+
                        os.path.join('./toy_label/sample', os.path.basename(p))+'txt' for p in filelist]
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            bdu.mark_prepared(self.root)