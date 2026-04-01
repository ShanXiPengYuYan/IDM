import numpy as np
import random
import os
from sklearn.metrics import confusion_matrix

def initialize_GPU(gpu):
    # Initialize GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

def mynorm(data, norm_type):
    data_norm = np.zeros(data.shape)
    if norm_type == 'bandwise':
        for i in range(data.shape[2]):
            data_max = np.max(data[:, :, i])
            data_min = np.min(data[:, :, i])
            data_norm[:, :, i] = (data[:, :, i] - data_min) / (data_max - data_min)
    elif norm_type == 'pixelwise':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_max = np.max(data[i, j, :])
                data_min = np.min(data[i, j, :])
                data_norm[i, j, :] = (data[i, j, :] - data_min) / (data_max - data_min)
    return data_norm

def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # 中心区域左边镜像
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    #
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    # 右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    # 上边镜像
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    # 下边镜像
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("Patch size: {}".format(patch))
    print("Padded image shape: [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    return mirror_hsi

# -------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image

def select_points(mask, num_classes, select_type, ratio=None, rngsd1=None):
    select_size = []
    select_pos = {}
    num_classes = int(num_classes)
    if select_type == 'normal':
        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask == (i + 1))
            select_size.append(each_class.shape[0])
            select_pos[i] = each_class

        total_select_pos = select_pos[0]
        for i in range(1, num_classes):
            total_select_pos = np.r_[total_select_pos, select_pos[i]]  # (695,2)
        total_select_pos = total_select_pos.astype(int)

    elif select_type == 'random':

        for i in range(num_classes):
            each_class = []
            each_class = np.argwhere(mask == (i + 1))
            lengthi = each_class.shape[0]
            num = range(1, lengthi)

            random.seed(rngsd1)
            nums = random.sample(num, int(lengthi * ratio))
            select_size.append(len(nums))
            select_pos[i] = each_class[nums, :]

    total_select_pos = select_pos[0]
    for i in range(1, num_classes):
        total_select_pos = np.r_[total_select_pos, select_pos[i]]  # (695,2)
    total_select_pos = total_select_pos.astype(int)

    return total_select_pos, select_size

# -------------------------------------------------------------------------------
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()
# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        # print(np.sum(matrix[i, :]))
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

# def cal_results(matrix):
#     shape = np.shape(matrix)
#     number = 0
#     sum = 0
#     AA = np.zeros([shape[0]], dtype=float)
#     for i in range(shape[0]):
#         number += matrix[i, i]
#         row_sum = np.sum(matrix[i, :])
#         if row_sum != 0:  # 防止除以零
#             AA[i] = matrix[i, i] / row_sum
#         else:
#             AA[i] = 1  # 或者可以设置为其他合理的值
#         sum += row_sum * np.sum(matrix[:, i])
#     OA = number / np.sum(matrix)
#     AA_mean = np.mean(AA)
#     pe = sum / (np.sum(matrix) ** 2)
#     Kappa = (OA - pe) / (1 - pe)
#     return OA, AA_mean, Kappa, AA