import csv
import math
import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional
import torchaudio
from PIL import Image
from scipy import signal
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from utils import *


class Mat_Dataset(Dataset):
    def __init__(self,  input_MultiModal, band_MultiModal, labels,  patch, select_type, rngsd1=None):
        '''

        :param folder_data:
        :param modals: eg:['data_HS_HR.mat', 'data_DSM_HR.mat'] or ['data_HS_LR.mat', 'data_SAR_HR.mat', 'data_DSM.mat']
        :param norm_type:
        :param patch:
        :param select_type:
        :param rngsd1:  random seed
        '''

        height, width, _ = input_MultiModal.shape

        self.labels = labels
        self.num_classes = np.max(labels)
        self.patch = patch
        self.total_pos_TR, self.number_TR = select_points(labels, self.num_classes, select_type, rngsd1)

        self.mirror_image = mirror_hsi(height, width, np.sum(band_MultiModal), input_MultiModal, patch=patch)

    def __getitem__(self, idx):
        data = gain_neighborhood_pixel(self.mirror_image, self.total_pos_TR, idx, patch=self.patch)

        label = self.labels[self.total_pos_TR[idx][0], self.total_pos_TR[idx][1]] - 1

        data = torch.FloatTensor(data.transpose(2, 0, 1))  # [# band, patch*patch]

        label = torch.LongTensor([label]).squeeze()


        # return data, label
        return data, label, idx

    def __len__(self):
        return len(self.total_pos_TR)



def load_dataset(dataset_name, norm_type):
    if dataset_name == 'Houston':
        folder_data = '../../data/HS-LiDAR Houston2013/'
        data_HS = loadmat(folder_data + 'data_HS_HR.mat')
        data_DSM1 = loadmat(folder_data + 'data_DSM_HR.mat')
        label_TR = loadmat(folder_data + 'TrainImage.mat')
        label_TE = loadmat(folder_data + 'TestImage.mat')

        input_HS = mynorm(data_HS['data_HS_HR'], norm_type)  # (349, 1905, 144)
        input_DSM1 = np.expand_dims(data_DSM1['DSM'], axis=-1)  # (349, 1905, 1)
        height, width, band1 = input_HS.shape
        _, _, band2 = input_DSM1.shape

        input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
        band_MultiModal = [band1, band2]
    elif dataset_name == 'Trento':
        folder_data = '../../data/HS-LiDAR Trento/'
        data_HS = loadmat(folder_data + 'HSI.mat')
        data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
        label_TR = loadmat(folder_data + 'TRLabel.mat')
        label_TE = loadmat(folder_data + 'TSLabel.mat')

        input_HS = mynorm(data_HS['HSI'], norm_type)  # (349, 1905, 144)

        input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)  # (349, 1905, 1)

        height, width, band1 = input_HS.shape
        _, _, band2 = input_DSM1.shape

        input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
        band_MultiModal = [band1, band2]
    elif dataset_name == 'Berlin':
        folder_data = '../../data/HS-SAR Berlin/'
        data_HS = loadmat(folder_data + 'data_HS_LR.mat')
        data_SAR = loadmat(folder_data + 'data_SAR_HR.mat')
        label_TR = loadmat(folder_data + 'TrainImage.mat')
        label_TE = loadmat(folder_data + 'TestImage.mat')

        input_HS = mynorm(data_HS['data_HS_LR'], norm_type)  # (349, 1905, 144)
        input_SAR = data_SAR['data_SAR_HR']  # (1723, 476, 4)

        height, width, band1 = input_HS.shape
        height_1, width_1, band2 = input_SAR.shape
        input_MultiModal = np.concatenate((input_HS, input_SAR), axis=2)
        band_MultiModal = [band1, band2]

    elif dataset_name == 'Augsburg':

        folder_data = '../../data/HS-SAR-DSM Augsburg/'
        data_HS = loadmat(folder_data + 'data_HS_LR.mat')
        data_SAR = loadmat(folder_data + 'data_SAR_HR.mat')
        data_DSM = loadmat(folder_data + 'data_DSM.mat')
        label_TR = loadmat(folder_data + 'TrainImage.mat')
        label_TE = loadmat(folder_data + 'TestImage.mat')

        input_HS = mynorm(data_HS['data_HS_LR'], norm_type)  # (349, 1905, 144)
        input_SAR = data_SAR['data_SAR_HR']  # (1723, 476, 4)
        input_DSM = np.expand_dims(data_DSM['data_DSM'], axis=-1)

        height, width, band1 = input_HS.shape

        _, _, band2 = input_SAR.shape
        _, _, band3 = input_DSM.shape

        input_MultiModal = np.concatenate((input_HS, input_SAR, input_DSM), axis=2)
        band_MultiModal = [band1, band2, band3]
        # input_MultiModal = np.concatenate((input_HS, input_SAR), axis=2)
        # band_MultiModal = [band1, band2]

    else:
        raise ValueError("Unknown dataset")
    if dataset_name == 'Trento':
        label_TR = label_TR['TRLabel']
        label_TE = label_TE['TSLabel']
    else:
        label_TR = label_TR['TrainImage']
        label_TE = label_TE['TestImage']

    return input_MultiModal, band_MultiModal, label_TR, label_TE



def get_dataloader(input_MultiModal, band_MultiModal,  label_TR,  patch, select_type, batch_size, num_workers,
                      distributed=False, rngsd1=None):


    dataset = Mat_Dataset(input_MultiModal=input_MultiModal, band_MultiModal=band_MultiModal, labels=label_TR,
                          patch=patch, select_type=select_type, rngsd1=None)
    if distributed:
    
        train_sampler = DistributedSampler(dataset, seed=rngsd1)
    else:
      
        train_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(rngsd1),  
        shuffle=True
    )
    return data_loader



def get_trian_val_loader(input_MultiModal, band_MultiModal,  label_TR,  patch, select_type, batch_size, num_workers,
                      distributed=False, rngsd1=None):


    dataset = Mat_Dataset( input_MultiModal=input_MultiModal, band_MultiModal=band_MultiModal, labels=label_TR,
                          patch=patch, select_type=select_type, rngsd1=None)

    
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size],
                                                               generator=torch.Generator().manual_seed(rngsd1))
                                                               # )
    if distributed:
  
        train_sampler = DistributedSampler(train_dataset, seed=rngsd1)
        val_sampler = DistributedSampler(val_dataset, seed=rngsd1)
    else:

        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.RandomSampler(val_dataset)
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        sampler=val_sampler
    )
    return train_loader, val_loader

def get_testloader(input_MultiModal, band_MultiModal, label_TE,  patch, select_type, batch_size, num_workers,
                       distributed=False, rngsd1=None):
    dataset = Mat_Dataset(input_MultiModal=input_MultiModal, band_MultiModal=band_MultiModal, labels=label_TE,
                          patch=patch, select_type=select_type, rngsd1=None)
    if distributed:

        test_sampler = DistributedSampler(dataset, seed=rngsd1)
    else:
 
        test_sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        generator=torch.Generator().manual_seed(rngsd1),  # 确保 DataLoader 洗牌的可复现性
    )
    return data_loader

# folder_data = '../data/HS-LiDAR Trento/'
# data_HS = loadmat(folder_data + 'HSI.mat')
# data_DSM1 = loadmat(folder_data + 'LiDAR.mat')
# label_TR = loadmat(folder_data + 'TRLabel.mat')
# label_TE = loadmat(folder_data + 'TSLabel.mat')
#
# print(label_TR)
# input_HS = mynorm(data_HS['HSI'], 'bandwise')  # (349, 1905, 144)
# input_DSM1 = np.expand_dims(data_DSM1['LiDAR'], axis=-1)  # (349, 1905, 1)

# height, width, band1 = input_HS.shape
# _, _, band2 = input_DSM1.shape
#
# input_MultiModal = np.concatenate((input_HS, input_DSM1), axis=2)
# band_MultiModal = [band1, band2]

# test
# initialize_GPU('0')
# norm_type = 'bandwise'  # 'pixelwise', 'bandwise'
# select_type = 'normal'
# batch_size = 64
# num_workers = 10
# rngsd = 1
# input_MultiModal, band_MultiModal, label_TR, label_TE = load_dataset('Houston', norm_type)
#
# data_loader = get_dataloader(input_MultiModal, band_MultiModal, label_TR,
#                 patch=13, select_type=select_type, batch_size=batch_size, num_workers=num_workers,
#                        distributed=False, rngsd1=rngsd)
# for batch_idx, (batch_data, batch_target) in enumerate(data_loader):
#     print(batch_idx)
#     print(batch_data.shape)
#     print(batch_target.shape)
