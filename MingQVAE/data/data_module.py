from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from tools.imp import instantiate_from_config

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size # 数据集调用 batch size
        self.dataset_configs = dict() # 将训练、验证、测试三个数据集的配置存入字典方便调用
        self.num_workers = num_workers if num_workers is not None else batch_size*2 # 未指定cpu数量时，使用batch size二倍
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        # ---------------------------------------
        #  实现数据集下载，单GPU执行
        #  依据配置文件分别读入三类数据集
        #  配置文件格式：target+params{config.size}
        # ---------------------------------------
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        # ----------------------------------------------
        #  实现数据集定义，多GPU执行，stage 用于标记是什么阶段
        # ----------------------------------------------
        self.datasets = dict()
        for k in self.dataset_configs.keys():
            self.datasets[k] = instantiate_from_config(self.dataset_configs[k])
            self.datasets[k].load()
            if self.wrap:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # -----------------------------------------------
    #  分别使用pytorch的DataLoader载入三个数据集(内部函数)
    #  配置文件
    # -----------------------------------------------
    def _train_dataloader(self): 
        return DataLoader(self.datasets["train"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)
                          