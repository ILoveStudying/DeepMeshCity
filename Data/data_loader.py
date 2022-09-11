from torch.utils.data import Dataset
import torch
import numpy as np
import os
import datetime as dt
import pandas as pd
import time


class Dataset_Bousai_CPT(Dataset):
    def __init__(self, configs, mode):
        self.datapath = configs.datapath
        self.datafile = configs.datafile
        self.datepath = configs.datepath
        self.max_value = configs.max_value
        self.trainRatio = configs.trainRatio
        self.validRatio = configs.validRatio
        self.TIMESTEP = configs.timestep
        self.DAYTIMESTEP = configs.daytimestep
        self.len_p = configs.perioid_len
        self.len_t = configs.trend_len
        self.DateStart = configs.DateStart
        self.DateEnd = configs.DateEnd
        self.freq = configs.freq
        self.train_data_paths = configs.train_data_paths
        self.valid_data_paths = configs.valid_data_paths
        self.test_data_paths = configs.test_data_paths
        self.city_type = configs.dataset_name + "_" + configs.type
        if mode == 'train':
            self.name = 'train_' + self.city_type + '.npz'
        elif mode == 'valid':
            self.name = 'valid_' + self.city_type + '.npz'
        else:
            self.name = 'test_' + self.city_type + '.npz'
        # self.get_data()
        path = os.path.join(self.datapath, self.name)
        if not os.path.exists(path):
            self.get_data()

        self.data = self.load_data(path)

    def get_data(self):
        data = np.load(self.datafile)
        self.build_metadate_information(self.DateStart, self.DateEnd, self.freq, self.datepath)
        dayinfo = np.genfromtxt(self.datepath, delimiter=',', skip_header=1)
        print('data.shape, dayinfo.shape', data.shape, dayinfo.shape)
        train_Num = int(data.shape[0] * self.trainRatio)

        print('training started', time.ctime())
        trainvalidateData = data[:train_Num, :, :, :]
        print('trainvalidateData.shape', trainvalidateData.shape)

        testData = data[train_Num:, :, :, :]
        print('testData.shape', testData.shape)

        XC, XP, XT, YS, YD = self.getXSYS_CPT_D('train', data, trainvalidateData, dayinfo)
        split_len = int(XC.shape[0] * (1 - self.validRatio))
        train_XC = XC[:split_len, :, :, :]
        train_XP = XP[:split_len, :, :, :]
        train_XT = XT[:split_len, :, :, :]
        train_YS = YS[:split_len, :, :, :]
        train_YD = YD[:split_len, :]

        valid_XC = XC[split_len:, :, :, :]
        valid_XP = XP[split_len:, :, :, :]
        valid_XT = XT[split_len:, :, :, :]
        valid_YS = YS[split_len:, :, :, :]
        valid_YD = YD[split_len:, :]

        print("train shape:", train_XC.shape, train_XP.shape, train_XT.shape, train_YS.shape, train_YD.shape)
        print("valid shape:", valid_XC.shape, valid_XP.shape, valid_XT.shape, valid_YS.shape, valid_YD.shape)

        test_XC, test_XP, test_XT, test_YS, test_YD = self.getXSYS_CPT_D('test', data, trainvalidateData, dayinfo)
        print("test shape:", test_XC.shape, test_XP.shape, test_XT.shape, test_YS.shape, test_YD.shape)

        train_data = {'xc': train_XC, 'xp': train_XP, 'xt': train_XT, 'ys': train_YS, 'yd': train_YD}
        valid_data = {'xc': valid_XC, 'xp': valid_XP, 'xt': valid_XT, 'ys': valid_YS, 'yd': valid_YD}
        test_Data = {'xc': test_XC, 'xp': test_XP, 'xt': test_XT, 'ys': test_YS, 'yd': test_YD}

        train_filename = os.path.join(self.datapath, 'train_' + self.city_type + '.npz')
        valid_filename = os.path.join(self.datapath, 'valid_' + self.city_type + '.npz')
        test_filename = os.path.join(self.datapath, 'test_' + self.city_type + '.npz')

        np.savez_compressed(train_filename, xc=train_data['xc'], xp=train_data['xp'], xt=train_data['xt'],
                            ys=train_data['ys'], yd=train_data['yd'])
        np.savez_compressed(valid_filename, xc=valid_data['xc'], xp=valid_data['xp'], xt=valid_data['xt'],
                            ys=valid_data['ys'], yd=valid_data['yd'])
        np.savez_compressed(test_filename, xc=test_Data['xc'], xp=test_Data['xp'], xt=test_Data['xt'],
                            ys=test_Data['ys'], yd=test_Data['yd'])

        print("data generated!")

    def build_metadate_information(self, START, END, freq, temporal_path):
        next_day = (dt.datetime.strptime(END, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
        date_information = pd.DataFrame({'datetime': pd.date_range(start=START, end=next_day, freq=freq)})
        date_information.drop([len(date_information) - 1], inplace=True)
        date_information['day'] = date_information.datetime.dt.dayofweek
        date_information['time'] = date_information.datetime.dt.time

        date_information['dayflag'] = 1
        date_information.loc[(date_information.day == 5) | (date_information.day == 6), 'dayflag'] = 0
        date_information.set_index('datetime', inplace=True)

        date_information = date_information.astype('str')
        date_information = pd.get_dummies(date_information)
        date_information.to_csv(temporal_path, index=False)
        print('temporal build finish')

    def getXSYS_CPT_D(self, mode, allData, trainData, dayinfo):
        len_c, len_p, len_t = self.TIMESTEP, self.len_p, self.len_t
        interval_p, interval_t = 1, 7

        stepC = list(range(1, len_c + 1))
        periods, trends = [interval_p * self.DAYTIMESTEP * i for i in range(1, len_p + 1)], \
                          [interval_t * self.DAYTIMESTEP * i for i in range(1, len_t + 1)]
        stepP, stepT = [], []
        for p in periods:
            stepP.extend(list(range(p, p + len_c)))
        for t in trends:
            stepT.extend(list(range(t, t + len_c)))
        depends = [stepC, stepP, stepT]

        if mode == 'train':
            start = max(stepT)
            end = trainData.shape[0]
        elif mode == 'test':
            start = trainData.shape[0] + len_c
            end = allData.shape[0]
        else:
            assert False, 'invalid mode...'

        XC, XP, XT, YS, YD = [], [], [], [], []
        for i in range(start, end):
            x_c = [allData[i - j][np.newaxis, :, :, :] for j in depends[0]]
            x_p = [allData[i - j][np.newaxis, :, :, :] for j in depends[1]]
            x_t = [allData[i - j][np.newaxis, :, :, :] for j in depends[2]]
            x_c = np.concatenate(x_c, axis=0)
            x_p = np.concatenate(x_p, axis=0)
            x_t = np.concatenate(x_t, axis=0)
            x_c = x_c[::-1, :, :, :]
            x_p = x_p[::-1, :, :, :]
            x_t = x_t[::-1, :, :, :]
            d = dayinfo[i]
            y = allData[i]
            XC.append(x_c)
            XP.append(x_p)
            XT.append(x_t)
            YS.append(y)
            YD.append(d)
        XC, XP, XT, YS, YD = np.array(XC), np.array(XP), np.array(XT), np.array(YS), np.array(YD)
        XC, XP, XT = list(map(lambda x: x.transpose(0, 1, 4, 2, 3), [XC, XP, XT]))
        YS = YS.transpose(0, 3, 1, 2)

        return XC, XP, XT, YS, YD

    def load_data(self, path):
        all_data = np.load(path)
        self.XC = all_data['xc']
        self.XP = all_data['xp']
        self.XT = all_data['xt']
        self.YS = all_data['ys']
        self.YD = all_data['yd']

        print(self.XC.shape, self.XP.shape, self.XT.shape, self.YS.shape, self.YD.shape)

        return all_data

    def __len__(self):
        return self.data['xc'].shape[0]

    def __getitem__(self, item):

        XC = torch.FloatTensor(self.XC[item])
        XP = torch.FloatTensor(self.XP[item])
        XT = torch.FloatTensor(self.XT[item])
        YS = torch.FloatTensor(self.YS[item])
        YD = torch.FloatTensor(self.YD[item])

        return XC, XP, XT, YS, YD


class Dataset_Taxibj(Dataset):
    def __init__(self, configs, mode):
        self.datapath = configs.datapath
        self.data_type = configs.data_type
        self.max_value = configs.max_value
        if mode == 'train':
            self.name = 'train_' + self.data_type + '.npz'
        elif mode == 'valid':
            self.name = 'valid_' + self.data_type + '.npz'
        else:
            self.name = 'test_' + self.data_type + '.npz'
        path = os.path.join(self.datapath, self.name)
        self.data = self.load_data(path)

    def load_data(self, path):
        all_data = np.load(path)
        self.XC = all_data['xc']
        self.XP = all_data['xp']
        self.XT = all_data['xt']
        self.YS = all_data['ys']
        self.YD = all_data['yd']

        print(self.XC.shape, self.XP.shape, self.XT.shape, self.YS.shape, self.YD.shape, self.max_value)

        return all_data

    def __len__(self):
        return self.data['xc'].shape[0]

    def __getitem__(self, item):
        XC = torch.FloatTensor(self.XC[item])
        XP = torch.FloatTensor(self.XP[item])
        XT = torch.FloatTensor(self.XT[item])
        YS = torch.FloatTensor(self.YS[item])
        YD = torch.FloatTensor(self.YD[item])

        return XC, XP, XT, YS, YD
