from torch.autograd import Variable
from tqdm import tqdm
import torch


def train_epoch(model, optimizer, loss_func, dataloader, lr_scheduler, epoch, device):
    loss_list = []
    model.train()
    print("The learning rate of the %dth epoch：%f" % (epoch, optimizer.param_groups[0]['lr']))
    for batch, item in tqdm(enumerate(dataloader)):
        xc, xp, xt, ys, yd = item
        xc, xp, xt, ys, yd = list(map(lambda x: Variable(x.to(device)), [xc, xp, xt, ys, yd]))
        optimizer.zero_grad()
        next_frame = model(xc, xp, xt, yd)
        loss = loss_func(next_frame, ys)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.detach().cpu().item())

    if lr_scheduler is not None:
        lr_scheduler.step(epoch)

    return sum(loss_list) / len(loss_list)


def test_epoch(model, loss_func, DataLoader, device):
    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch, item in tqdm(enumerate(DataLoader)):
            xc, xp, xt, ys, yd = item
            xc, xp, xt, ys, yd = list(map(lambda x: x.to(device), [xc, xp, xt, ys, yd]))
            next_frame = model(xc, xp, xt, yd)

            loss = loss_func(next_frame, ys)
            loss_list.append(loss.detach().cpu().item())

    return sum(loss_list) / len(loss_list)
