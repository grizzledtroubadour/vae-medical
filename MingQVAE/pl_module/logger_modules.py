from omegaconf import OmegaConf
from tools.imp import instantiate_from_config

#  获取checkpoint类
def get_logger(logdir, logger_config=None):
	'''
	输入：
	—————————————————————————————————————————————————————————
	logdir：log信息保存目录
	logger_config：logger配置信息，OmegaConf封装，格式同下默认配置
	—————————————————————————————————————————————————————————
	返回：
	pytorch_lightning的logger对象
	'''

	# 默认的logger配置，TestTubeLogger是格式简单的TensorBoardLogger
	default_logger_config = {
		'target': "pytorch_lightning.loggers.TestTubeLogger",
		'params': {
			'name': "testtube",
			'save_dir': logdir,
		}
	}

	# OmegaConf.merge()函数第二个字典的相同同名变量会覆盖第一个
	logger_config = OmegaConf.merge(default_logger_config, logger_config)
	return instantiate_from_config(logger_config)
	