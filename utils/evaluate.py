import datetime
import seaborn as sns
import numpy as np
import os.path
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def evaluate_model(model, MSE_criterion, DataLoader, device, configs, itr=999):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    if not os.path.exists(res_path):
        os.mkdir(res_path)
    avg_mse = 0
    avg_rmse = 0
    avg_mae = 0
    batch_id = 0

    loss_list = []
    model.eval()
    with torch.no_grad():
        for batch, item in enumerate(DataLoader):
            batch_id = batch_id + 1
            xc, xp, xt, ys, yd = item
            xc, xp, xt, ys, yd = list(map(lambda x: x.to(device), [xc, xp, xt, ys, yd]))
            pred = model(xc, xp, xt, yd)
            loss = MSE_criterion(pred, ys)
            loss_list.append(loss.cpu().item())

            # MSE per frame
            x = ys.cpu().numpy() * configs.max_value  # (B, H, W, C)
            gx = pred.cpu().numpy() * configs.max_value
            gx[gx < 1] = 0

            mae = np.mean(np.abs(x - gx))
            mse = np.mean(np.square(x - gx))
            rmse = np.mean(np.square(x - gx))

            avg_mae += mae
            avg_mse += mse
            avg_rmse += rmse

            # save prediction examples
            if batch_id <= configs.num_save_samples:
                path = os.path.join(res_path, str(batch_id))
                if not os.path.exists(path):
                    os.mkdir(path)
                for i in range(configs.input_length):
                    name = 'gt' + str(i + 1) + '.png'
                    file_name = os.path.join(path, name)
                    img_c = xc[0, i, 0, :, :].cpu().numpy()
                    draw_pic(img_c, file_name)

                name = 'pd7' + '.png'
                file_name = os.path.join(path, name)
                img_pd = pred[0, 0, :, :].cpu().numpy()
                draw_pic(img_pd, file_name)

                name = 'gt7' + '.png'
                file_name = os.path.join(path, name)
                img_gt = ys[0, 0, :, :].cpu().numpy()
                draw_pic(img_gt, file_name)

        avg_mse = avg_mse / batch_id
        avg_mae = avg_mae / batch_id
        avg_rmse = np.sqrt(avg_rmse / batch_id)
        print('mse per frame: {}, rmse per frame: {}, mae per frame: {}'.format(avg_mse, avg_rmse, avg_mae))

    return avg_mse, avg_rmse, avg_mae


def draw_pic(img_gt, file_name):
    img_gt[img_gt > 0.2] = 0.2
    factor = 1.0 / 3.0
    img_gt[(img_gt > 0.05) & (img_gt <= 0.2)] = img_gt[(img_gt > 0.05) & (img_gt <= 0.2)] * factor + 2 * factor * 0.05

    fig = sns.heatmap(img_gt, annot=False, vmin=0, vmax=0.1, xticklabels=False,
                      yticklabels=False, cbar=False, cmap='rainbow', square=True)
    heatmap = fig.get_figure()
    heatmap.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.clf()