import torch
from torch.utils.data import DataLoader
from Data import data_handle
import torch.nn as nn
from utils import evaluate
import datetime
from Model import model_factory

def test_main(cfg):
    # config
    test_cfg = cfg.test_cfg
    dataset_cfg = cfg.dataset_cfg
    device = cfg.device
    model_cfg = cfg.model_cfg

    # dataset
    test_dataset = data_handle.select_data(dataset_cfg, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=test_cfg.batch_size, shuffle=False, drop_last=True)

    # loss
    model = model_factory.build_model(model_cfg).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    MSE_criterion = nn.MSELoss().to(device)
    print(test_cfg.check_point_file)
    checkpoint = torch.load(test_cfg.check_point_file, map_location=device)
    model.load_state_dict(checkpoint.state_dict())

    test_loss = evaluate.evaluate_model(model, MSE_criterion, test_loader, device, test_cfg)
    print('The loss on test set is {:.6f}'.format(test_loss[0]))

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')