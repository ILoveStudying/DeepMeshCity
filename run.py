import os
import torch
import numpy as np
import random
import sys
from utils import config, trainer, tester


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    cfg_filename = sys.argv[1]  # config filename
    data = sys.argv[2]  # select osaka
    task = sys.argv[3]  # flow,density
    mode = sys.argv[4]  # mode: train,test

    # read config
    cfg = config.Config.fromfile('./Config/' + data + '/' + task + '/' + cfg_filename)
    print("config filename: " + str(cfg_filename))

    set_seed(cfg.random_seed)
    print("random seed is {}".format(cfg.random_seed))

    split_save_dir = cfg.train_cfg.check_point_file.split('/')
    save_dir = os.path.join(split_save_dir[0], os.path.join(split_save_dir[1], split_save_dir[2]))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(cfg.train_cfg.gen_frm_dir):
        os.makedirs(cfg.train_cfg.gen_frm_dir)

    # train model
    if mode == 'train':
        trainer.train_main(cfg=cfg)

    # test model
    if mode == 'test':
        tester.test_main(cfg=cfg)
