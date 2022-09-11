import os
from torch.utils.data import DataLoader
from Data import data_handle
from torch import optim
from Model import model_factory
from utils import evaluate
import time
import torch.nn as nn
import numpy as np
from utils.trainer_epoch import *


def train_main(cfg):
    # config
    train_cfg = cfg.train_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg
    device = cfg.device

    # dataset
    train_dataset = data_handle.select_data(dataset_cfg, mode='train')
    val_dataset = data_handle.select_data(dataset_cfg, mode='valid')
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, pin_memory=True,
                            drop_last=True)

    print(train_cfg.optimizer_cfg)
    # build model
    model = model_factory.build_model(model_cfg).to(device)

    # whether Parallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    parameters = model.parameters()

    # Parameter information
    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
        total_param += np.prod(model.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    # loss
    MSE_criterion = nn.MSELoss().to(device)

    # optimizer
    optimizer_cfg = train_cfg.optimizer_cfg
    lr_scheduler_cfg = train_cfg.lr_scheduler_cfg
    if optimizer_cfg.type == 'adam':
        optimizer = optim.Adam(params=parameters,
                               lr=optimizer_cfg.lr)
    elif optimizer_cfg.type == 'adamw':
        optimizer = optim.AdamW(params=parameters,
                                lr=optimizer_cfg.lr,
                                weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'sgd':
        optimizer = optim.SGD(params=parameters,
                              lr=optimizer_cfg.lr,
                              momentum=optimizer_cfg.momentum,
                              weight_decay=optimizer_cfg.weight_decay)
    elif optimizer_cfg.type == 'RMS':
        optimizer = optim.RMSprop(params=parameters,
                                  lr=optimizer_cfg.lr,
                                  momentum=optimizer_cfg.momentum,
                                  weight_decay=optimizer_cfg.weight_decay)
    else:
        raise Exception('No Optimizer！')

    # learning schedule
    if lr_scheduler_cfg is None:
        lr_scheduler = None
    elif lr_scheduler_cfg.policy == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_scheduler_cfg.step_size,
                                                       gamma=lr_scheduler_cfg.gamma, last_epoch=-1)
    elif lr_scheduler_cfg.policy == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, lr_scheduler_cfg.T_0,
                                                                            lr_scheduler_cfg.T_mult,
                                                                            lr_scheduler_cfg.eta_min)
    else:
        lr_scheduler = None

    train_loss_list = []
    val_loss_list = []
    Best_Metric = 0
    Min_Metric = 999999999
    best_epoch = 0
    check_point_dir = '/'.join(train_cfg.check_point_file.split('/')[:-1])

    if not os.path.exists(check_point_dir):
        os.mkdir(check_point_dir)

    for epoch in range(1, train_cfg.num_epochs + 1):
        print()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        start_time = time.time()
        print(f"It is the {epoch}th epoch...")

        # training
        train_loss = train_epoch(model, optimizer, MSE_criterion, train_loader, lr_scheduler, epoch, device)
        if epoch % train_cfg.test_interval == 0:
            # validate
            val_loss = evaluate.evaluate_model(model, MSE_criterion, val_loader, device, train_cfg, epoch)
            Best_Metric = val_loss[0]

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # save model
        if Min_Metric > Best_Metric:
            Min_Metric = Best_Metric
            best_epoch = epoch
            torch.save(model, train_cfg.check_point_file)

        if epoch == 75 or epoch == 155:
            model_file = train_cfg.check_point_file.split('.')[0] + '-epoch{}.pth'.format(epoch)
            torch.save(model, model_file)

        # print out
        test_end_time = time.time()
        run_time = int(test_end_time - start_time)
        m, s = divmod(run_time, 60)
        time_str = "{:02d}m{:02d}s".format(m, s)
        out_str = "The {} epoch is finished, consuming {},\n" \
                  "The loss on training set is {:.6f}；the best epoch is {}, best_metric={:.6f}" \
            .format(epoch, time_str, sum(train_loss_list) / len(train_loss_list), best_epoch, Min_Metric)
        print(out_str)
