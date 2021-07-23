import importlib


# 从字符串指定路径位置输入类或函数对象
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    module_imp = importlib.import_module(module, package=None) # import指定路径的文件化为对象导入
    if reload:
        importlib.reload(module_imp) # 在运行过程中若修改了库，需要使用reload重新载入
    return getattr(module_imp, cls) # getattr()函数获取对象中对应字符串的对象属性（可以是值、函数等）

# 从配置中载入模型
def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module = get_obj_from_str(config["target"]) # target路径的类或函数模块
    params_config = config.get('params', dict()) # 对应模块的参数配置
    return module(**params_config)