import os, tarfile, glob, shutil, pickle
from torch.utils.data.dataset import Dataset
import yaml
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from tools.utils import download, retrieve
import data.utils as bdu

class CIFAR10Base(Dataset):
	NAME = "CIFAR10"
	FILES = "cifar-10-python.tar.gz"
	SIZES = 170498071

	def __init__(self, config=None):
		self.config = config or OmegaConf.create()
		if not type(self.config) == dict:
			self.config = OmegaConf.to_container(self.config)
		self._prepare()

	def _prepare(self):
		cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
		self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
		self.datadir = os.path.join(self.root, 'data')
		
		# --------------
        #  自动解压数据集
        # --------------
		if not bdu.is_prepared(self.root):
			print("Preparing dataset {} in {}".format(self.NAME, self.root))

			datadir = self.datadir
			if not os.path.exists(datadir):
				path = os.path.join(self.root, self.FILES)
				print("Extracting {} to {}".format(path, datadir))
				os.makedirs(datadir, exist_ok=True)
				with tarfile.open(path, "r:gz") as tar:
					tar.extractall(path=datadir)

			bdu.mark_prepared(self.root)

	def load(self):
		raise NotImplementedError()

	def __len__(self):
		return self._length

	def __getitem__(self, i):
		example = dict()
		example["image"] = (self.data['data'][i]/127.5 - 1.0).astype(np.float32)
		example["label"] = self.data['labels'][i]
		#return example["image"], example["label"]
		return example

class CIFAR10Train(CIFAR10Base):
	BATCHES = [
		"cifar-10-batches-py/data_batch_1",
		"cifar-10-batches-py/data_batch_2",
		"cifar-10-batches-py/data_batch_3",
		"cifar-10-batches-py/data_batch_4",
		"cifar-10-batches-py/data_batch_5",
	]

	def load(self):
		self.data = {
			'data': np.empty((0,32,32,3)),
			'labels': []
		}

		for i, batch_name in enumerate(self.BATCHES):
			file = os.path.join(self.datadir, batch_name)
			with open(file, 'rb') as fo:
				batch_data = pickle.load(fo, encoding='bytes')
			# dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
			self.data['data'] = np.concatenate((self.data['data'], batch_data[b'data'].reshape([10000,3,32,32]).transpose(0,2,3,1)), axis=0)
			self.data['labels'] += batch_data[b'labels']

		self._length = len(self.data['labels'])
		print('CIFAR-10 train load finish.')

class CIFAR10Validation(CIFAR10Base):
	BATCHES = [
		"cifar-10-batches-py/test_batch",
	]

	def load(self):
		self.data = {
			'data': np.empty((0,32,32,3)),
			'labels': []
		}

		for i, batch_name in enumerate(self.BATCHES):
			file = os.path.join(self.datadir, batch_name)
			with open(file, 'rb') as fo:
				batch_data = pickle.load(fo, encoding='bytes')
			self.data['data'] = np.concatenate((self.data['data'], batch_data[b'data'].reshape([10000,3,32,32]).transpose(0,2,3,1)), axis=0)
			self.data['labels'] += batch_data[b'labels']

		self._length = len(self.data['labels'])
		print('CIFAR-10 validation load finish.')


