# import numpy as np
# from scipy.spatial.distance import cdist
#
# # 样本数据
# X = np.array([[0, 0.5], [-0.5, 1], [2, 4.5], [3.5, 1.5], [4.5, 2.5]])
#
# # 初始聚类中心
# mu1 = X[0]
# mu2 = X[1]
#
# # 协方差矩阵
# S = np.cov(X.T)
#
# # 迭代次数
# max_iter = 10
#
# for i in range(max_iter):
#     # 计算每个样本到各个聚类中心的马氏距离
#     D1 = cdist(X, mu1.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))
#     D2 = cdist(X, mu2.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))
#
#     # 将样本分配到最近的聚类中心所在的类别
#     classes = np.argmin(np.concatenate([D1, D2], axis=1), axis=1)
#
#     # 更新聚类中心
#     mu1_new = np.mean(X[classes==0], axis=0)
#     mu2_new = np.mean(X[classes==1], axis=0)
#
#     # 打印当前迭代的结果
#     print('迭代次数:', i+1)
#     for j in range(X.shape[0]):
#         if classes[j] == 0:
#             dist1 = cdist(X[j].reshape(1,-1), mu1.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))[0][0]
#             dist2 = cdist(X[j].reshape(1,-1), mu2.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))[0][0]
#             print('样本X{}到聚类中心μ1{}的马氏距离为{:.2f}，到聚类中心μ2{}的马氏距离为{:.2f}，因此分配到第1类。'.format(j+1, mu1, dist1, mu2, dist2))
#         else:
#             dist1 = cdist(X[j].reshape(1,-1), mu1_new.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))[0][0]
#             dist2 = cdist(X[j].reshape(1,-1), mu2_new.reshape(1,-1), 'mahalanobis', VI=np.linalg.inv(S))[0][0]
#             print('样本X{}到聚类中心μ1{}的马氏距离为{:.2f}，到聚类中心μ2{}的马氏距离为{:.2f}，因此分配到第2类。'.format(j+1, mu1_new, dist1, mu2_new, dist2))
#
#     # 更新聚类中心
#     mu1 = mu1_new
#     mu2 = mu2_new
#
# # 最终聚类结果
# cluster1_center = mu1
# cluster1_samples = X[classes==0]
# cluster2_center = mu2
# cluster2_samples = X[classes==1]
#
# # 打印最终聚类结果
# print('最终聚类结果：')
# print('第1类：')
# print('聚类中心：', cluster1_center)
# print('包含的样本：', cluster1_samples)
# print('第2类：')
# print('聚类中心：', cluster2_center)
# print('包含的样本：', cluster2_samples)

import numpy as np

# 定义样本数据
w1 = np.array([[1, 1.2], [2, 2.2], [3, 3.5]])
w2 = np.array([[1.1, 1], [1.9, 1.4], [3.5, 3.1]])

# 计算均值
mean_w1 = np.mean(w1, axis=0)
mean_w2 = np.mean(w2, axis=0)

# 计算协方差矩阵
S1 = np.cov(w1.T)
S2 = np.cov(w2.T)
Sw = (S1 + S2) / 2

# 求解参数a
a = np.dot(np.linalg.inv(Sw), (mean_w1 - mean_w2))
a1, a2 = a
a3 = -(a1*mean_w1[0] + a2*mean_w1[1])

# 打印结果
print("Classification plane: g(x) = {:.4f}*x1 + {:.4f}*x2 + {:.4f}".format(a1, a2, a3))
