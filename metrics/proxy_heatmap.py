
import copy
import contextlib
import sys


sys.path.insert(0, '..')

import faiss
import numpy as np
from sklearn.preprocessing import normalize
import time
import torch
from tqdm import tqdm

from metrics import e_recall, c_recall, dists, rho_spectrum
from metrics import nmi, f1, mAP_1000, c_mAP_1000
from metrics import coding_rate
from utilities import misc

#绘图
import matplotlib.pyplot as plt
import os
import datetime
import matplotlib as mpl
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def plot_similarity_matrix(features, epoch, save_path):
    print('ploting_heatmap..........')
    # 初始化绘图设置
    mpl.rcParams.update(mpl.rcParamsDefault)
    # 计算相似度矩阵
    similarity_matrix = np.dot(features, features.T)

    # 绘制画布
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制相似度矩阵
    plt.imshow(similarity_matrix, cmap='YlGn', vmin=0, vmax=1)
    # im1 = axes[0].imshow(similarity_matrix, cmap='YlGn', vmin=0, vmax=1)
    # 设置坐标轴
    xticks = [0,features.shape[0]]
    yticks = [0,features.shape[0]]
    # class_indices_end = []
    # for label in unique_labels:
    #     indices_end = np.where(labels == label)[0]
    #     class_indices_end.append(indices_end[-1])  # 取类别样本的最后一个位置索引
    # ept_lable = ['']*len(class_indices_end)
    # xticks = class_indices_end
    # yticks = class_indices_end

    # plt.xticks(xticks,ept_lable)
    # plt.yticks(yticks,ept_lable)

    # 添加颜色条
    plt.colorbar()
    # cbar = fig.colorbar(im1, ax=axes[0])


    # num_classes_x = len(unique_labels)
    # class_indices_x = []
    # for label in unique_labels:
    #     indices_x = np.where(labels == label)[0]
    #     class_indices_x.append((indices_x.min() + indices_x.max()) // 2)  # 取类别样本的中间位置索引
    #
    # for idx_x, class_idx_x in enumerate(class_indices_x):
    #     plt.text(class_idx_x, features.shape[0] + 5, str(unique_labels[idx_x]), ha='center', va='top')
    #
    # num_classes_y = len(unique_labels)
    # class_indices_y = []
    # for label in unique_labels:
    #     indices_y = np.where(labels == label)[0]
    #     class_indices_y.append((indices_y.min() + indices_y.max()) // 2)  # 取类别样本的中间位置索引
    #
    # for idx_y, class_idx_y in enumerate(class_indices_y):
    #     # plt.text(features.shape[1] + 5, class_idx_y, str(unique_labels[idx_y]), ha='right', va='center')
    #     plt.text(-7, class_idx_y, str(unique_labels[idx_y]), ha='right', va='center')




    # mpl.rcParams.update(mpl.rcParamsDefault)
    # 重置Matplotlib参数设置
    # 保存图像并清除窗口
    folder_path = os.path.join(save_path, 'P_Heat_map')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filenames = os.listdir(folder_path)
    num_files = len(filenames)
    file_path1 = os.path.join(folder_path, 'epoch_'+str(epoch) + '_proxy_heat_map.png')
    file_path2 = os.path.join(folder_path, 'epoch_' + str(epoch) + '_proxy_heat_map.pdf')
    plt.savefig(file_path1, dpi=200)
    plt.savefig(file_path2, dpi=200)
    plt.clf()
    plt.close()