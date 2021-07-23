from omegaconf import OmegaConf
from tools.imp import instantiate_from_config

#  获取checkpoint类
def get_checkpoint_callback(checkpoint_dir, ckpt_config=None, model_monitor=None):
	'''
	输入：
	———————————————————————————————————————————————————————————
	checkpoint_dir：checkpoint保存目录
	ckpt_config：checkpoint配置信息，OmegaConf存储，格式同下默认配置
	———————————————————————————————————————————————————————————
	返回：
	pytorch_lightning的checkpoint callback对象
	'''

	# 默认的checkpoint配置
	default_ckpt_config = {
		"target": "pytorch_lightning.callbacks.ModelCheckpoint",
		"params": {
			'dirpath': checkpoint_dir,
			'filename': '{epoch:06}',
			'verbose': True, # 冗余模式，不明白什么意思
			'save_last': True,
		}
	}
	# 如果存在监控指标，默认设置多个保存模型
	if model_monitor:
		print(f"Monitoring {model_monitor} as checkpoint metric.")
		default_ckpt_config['params']['monitor'] = model_monitor
		default_ckpt_config['params']['save_top_k'] = 3

	# OmegaConf.merge()函数第二个字典的相同同名变量会覆盖第一个
	ckpt_config = OmegaConf.merge(default_ckpt_config, ckpt_config)
	return instantiate_from_config(ckpt_config)
	