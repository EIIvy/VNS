import importlib
from IPython import embed
import os
import time
import yaml
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
import time
def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'model.TransE'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def save_config(args):
    args.save_config = False  #防止和load_config冲突，导致把加载的config又保存了一遍
    if not os.path.exists("config"):
        os.mkdir("config")
    config_file_name = time.strftime(str(args.model_name)+"_"+str(args.dataset_name)) + ".yaml"
    day_name = time.strftime("%Y-%m-%d")
    if not os.path.exists(os.path.join("config", day_name)):
        os.makedirs(os.path.join("config", day_name))
    config = vars(args)
    with open(os.path.join(os.path.join("config", day_name), config_file_name), "w") as file:
        file.write(yaml.dump(config))

def load_config(args, config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)
    return args

def get_param(*shape):
    param = Parameter(torch.zeros(shape))
    xavier_normal_(param)
    return param 

import logging


def get_logger(args):
# 第一步，创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关  此时是INFO

    # 第二步，创建一个handler，用于写入日志文件
    exp_lodder = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    folder = args.logger_path+args.model_name
    if not os.path.exists(folder):
        os.makedirs(folder)
    logfile =args.dataset_name+'_'+str(args.IMG)+'_'+exp_lodder+'_log.txt'
    fh = logging.FileHandler(folder+'/'+logfile, mode='a')  # open的打开模式这里可以进行参考
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 第三步，再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)   # 输出到console的log等级的开关

    # 第四步，定义handler的输出格式（时间，文件，行数，错误级别，错误提示）
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # 第五步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(ch)    

    return logger