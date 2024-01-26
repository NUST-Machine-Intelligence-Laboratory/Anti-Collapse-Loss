from scipy.spatial import distance
from sklearn.preprocessing import normalize
import numpy as np
import torch


class Metric():
    def __init__(self, mode, **kwargs):
        self.mode = mode
        self.requires = ['embeds', 'target_labels']
        self.name = 'dists@{}'.format(mode)

    def __call__(self, embeds, target_labels):
        embeds = embeds[:5000, :]
        target_labels = target_labels[:5000, :]
        features_locs = []
        # embeds = embeds[:90, :]
        # target_labels = target_labels[:90, :]
        #获得同类别样本的索引用于寻找特征 CUB 为类别数*单类别样本数的列表
        for lab in np.unique(target_labels):
            features_locs.append(np.where(target_labels == lab)[0])
            # print(np.size(np.where(target_labels == lab)[0]))
        # print('features_locs',len(features_locs))

        if 'intra' in self.mode:
            if isinstance(embeds, torch.Tensor):
                intrafeatures = embeds.detach().cpu().numpy()
            else:
                intrafeatures = embeds

            intra_dists = []

            for loc in features_locs:
                #类内样本距离矩阵并合并
                c_dists = distance.cdist(intrafeatures[loc],
                                         intrafeatures[loc], 'cosine')
                # c_dists = np.sum(c_dists) / (len(c_dists)**2 - len(c_dists))
                c_dists = np.sum(c_dists) / (len(c_dists) ** 2 - len(c_dists))
                intra_dists.append(c_dists)
            intra_dists = np.array(intra_dists)

            #处理距离矩阵 `intra_dists`，将其中的 NaN 值和无穷大值替换为最大值，并计算距离矩阵的均值。
            maxval = np.max(intra_dists[1 - np.isnan(intra_dists)])
            intra_dists[np.isnan(intra_dists)] = maxval
            intra_dists[np.isinf(intra_dists)] = maxval
            dist_metric = dist_metric_intra = np.mean(intra_dists)

        # if 'inter' in self.mode:
        #     if not isinstance(embeds, torch.Tensor):
        #         coms = []
        #         for loc in features_locs:
        #             com = normalize(
        #                 np.mean(embeds[loc], axis=0).reshape(1,
        #                                                      -1)).reshape(-1)
        #             coms.append(com)
        #         mean_inter_dist = distance.cdist(np.array(coms),
        #                                          np.array(coms), 'cosine')
        #         dist_metric = dist_metric_inter = np.sum(mean_inter_dist) / (
        #             len(mean_inter_dist)**2 - len(mean_inter_dist))
        #
        #     else:
        #         coms = []
        #         for loc in features_locs:
        #             # com为1*128的均值归一化特征
        #             com = torch.nn.functional.normalize(torch.mean(
        #                 embeds[loc], dim=0).reshape(1, -1), dim=-1).reshape(1, -1)
        #             coms.append(com)
        #         mean_inter_dist = 1 - torch.cat(coms, dim=0).mm(
        #             torch.cat(coms, dim=0).T).detach().cpu().numpy()
        #         dist_metric = dist_metric_inter = np.sum(mean_inter_dist) / (
        #             len(mean_inter_dist)**2 - len(mean_inter_dist))
        if 'inter' in self.mode:
            ## 快速均值(不准)
            # unique_labels = np.unique(target_labels)  # 获取唯一的标签值
            # num_classes = len(unique_labels)  # 类别数目
            #
            # distances = distance.cdist(embeds.cpu(), embeds.cpu(), metric='cosine')  # 计算余弦距离矩阵
            #
            # class_distances = []
            # for i in range(num_classes):
            #     class_i_indices = np.where(target_labels == unique_labels[i])[0]  # 属于类别 i 的样本索引
            #
            #     class_i_distances = distances[class_i_indices][:, class_i_indices]  # 类别 i 内部样本的余弦距离子矩阵
            #     class_i_mean_distance = np.mean(class_i_distances)  # 类别 i 内部样本之间的平均余弦距离
            #     class_distances.append(class_i_mean_distance)
            # 全特征均值(准)
            unique_labels = np.unique(target_labels)  # 获取唯一的标签值
            num_classes = len(unique_labels)  # 类别数目

            class_distances = []
            for i in range(num_classes):
                class_i_indices = np.where(target_labels == unique_labels[i])[0]  # 属于类别 i 的样本索引
                class_i_features = embeds[class_i_indices]  # 属于类别 i 的样本特征

                other_class_indices = np.where(target_labels != unique_labels[i])[0]  # 不属于类别 i 的样本索引
                other_class_features = embeds[other_class_indices]  # 不属于类别 i 的样本特征

                # 计算类别 i 和其他类别之间的余弦距离
                cosine_distances = distance.cdist(class_i_features.cpu(), other_class_features.cpu(), metric='cosine')
                average_cosine_distance = np.mean(cosine_distances)
                class_distances.append(average_cosine_distance)

            dist_metric = dist_metric_inter = np.mean(class_distances)  # 计算类别之间的平均余弦距离



        if self.mode == 'intra_over_inter':
            dist_metric = dist_metric_intra / np.clip(dist_metric_inter, 1e-8,
                                                      None)

        return dist_metric
