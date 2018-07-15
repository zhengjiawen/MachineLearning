import numpy as np
from unsupervised import kmeans
import matplotlib.pyplot as plt

def bimeans(dataSet, k):
    m, n = np.shape(dataSet)
    # 起始时，只有一个簇，该簇的聚类中心为所有样本的平均位置
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    # 设置一个列表保存当前的聚类中心
    currentCentroids = [centroid0]
    # 点分配结果： 第一列指明样本所在的簇，第二列指明该样本到聚类中心的距离
    clusterAssign = np.mat(np.zeros((m, 2)))
    # 初始化点分配结果，默认将所有样本先分配到初始簇
    for j in range(m):
        clusterAssign[j, 1] = kmeans.calEclud(dataSet[j, :], np.mat(centroid0)) ** 2

    minCost = np.inf
    # 直到簇的数目达标
    while len(currentCentroids) < k:
        # 当前最小的代价
        lowestError = np.inf
        # 对于每一个簇
        for j in range(len(currentCentroids)):
            # 获得该簇的样本
            ptsInCluster = dataSet[np.nonzero(clusterAssign[:, 0].A == j)[0], :]
            # if len(ptsInCluster) == 0:
            #     currentCentroids.pop(j)
            #     j-=1
            #     continue;
            # 在该簇上进行2-means聚类
            # 注意，得到的centroids，其聚类编号含0，1
            centroids, clusterAss ,splitedError= kmeans.kMeans(ptsInCluster, 2, 10, 1)
            # 获得划分后的误差之和
            # splitedError = np.sum(clusterAss[:, 1])
            # 获得其他簇的样本
            ptsNoInCluster = dataSet[np.nonzero(
                clusterAssign[:, 0].A != j)[0]]
            # 获得剩余数据集的误差
            nonSplitedError = np.sum(ptsNoInCluster[:, 1])
            # 比较，判断此次划分是否划算
            if (splitedError + nonSplitedError) < lowestError:
                # 如果划算，刷新总误差
                lowestError = splitedError + nonSplitedError
                # 记录当前的应当划分的簇
                needToSplit = j
                # 新获得的簇以及点分配结果
                newCentroids = centroids.A
                newClusterAss = clusterAss.copy()
        print("cost:" + str(lowestError))
        # 更新簇的分配结果
        # 第0簇应当修正为被划分的簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 0)[
                          0], 0] = needToSplit
        # 第1簇应当修正为最新一簇
        newClusterAss[np.nonzero(newClusterAss[:, 0].A == 1)[
                          0], 0] = len(currentCentroids)
        # 被划分的簇需要更新
        currentCentroids[needToSplit] = newCentroids[0, :]
        # 加入新的划分后的簇
        currentCentroids.append(newCentroids[1, :])
        # 刷新点分配结果
        clusterAssign[np.nonzero(
            clusterAssign[:, 0].A == needToSplit
        )[0], :] = newClusterAss
    return np.mat(currentCentroids), clusterAssign

if __name__ == "__main__":
    dataMat = np.mat(kmeans.loadDataSet("data\ex7data2.mat"))
    centroids, clusterAssment = bimeans(dataMat, 3)
    print("=======================")
    print(clusterAssment)
    clusterCount = np.shape(centroids)[0]
    m = np.shape(dataMat)[0]
    # 绘制散点图
    patterns = ['o', 'D', '^', 's']
    colors = ['b', 'g', 'y', 'black']
    fig = plt.figure()
    title = 'unsupervised with k=3'
    ax = fig.add_subplot(111, title=title)
    for k in range(clusterCount):
        # 绘制聚类中心
        ax.scatter(centroids[k, 0], centroids[k, 1], color='r', marker='+', linewidth=20)
        for i in range(m):
            # 绘制属于该聚类中心的样本
            ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A==k)[0]]
            ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], marker=patterns[k], color=colors[k])
    plt.show()