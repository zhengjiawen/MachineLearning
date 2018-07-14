import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def loadDataSet(filepath):
    '''
    加载数据
    :param filepath: 路径
    :return: 数据集，每个数据都是一个向量
    '''
    data = loadmat(filepath)
    X = np.array(data.get("X"))
#    print(X)
    return X

def calEclud(x,y):
    '''
    计算欧式距离
    :param x: 第一个向量
    :param y: 第二个向量
    :return: 两个向量的欧氏距离
    '''
    return np.sqrt(np.sum(np.power(x-y,2)))

def cost(clusterAssign):
    '''
    计算损失函数
    :param clusterAssign: 记录点被分配的簇和距离
    :return: 
    '''
    cost = np.sum(clusterAssign, axis=0).squeeze()
    return cost[0,1]

def generateRandCenterOfCluster(dataSet, k):
    '''
    随机生成k个聚类中心
    :param dataSet: 训练集
    :param k: 聚类中心数量
    :return: 
        centerCluster: 生成的聚类中心
    '''
    n = dataSet.shape[1]
    #这里直接使用np.array的话，下面生成centerCluster会报错，不能进行广播，需要转换成矩阵
    centerCluster = np.mat(np.zeros((k,n)))

    for j in range(n):
        minj = np.min(dataSet[:,j])
        maxj = np.max(dataSet[:,j])
        rangej = float(maxj - minj)
        centerCluster[:,j] = minj + rangej * np.random.rand(k, 1)
    return centerCluster

def kMeans(dataSet, k, maxIter = 5, maxRandCenterIter = 5):
    '''
    k-means model
    :param dataSet: 数据集 
    :param k: 聚类数
    :param maxIter:最大迭代次数 
    :param maxRandCenterIter:最大初始化次数 
    :return: 
        centerCluster: 聚类中心
        clusterAssign: 点分配结果
    '''


    m, n = dataSet.shape



    minCost = np.inf
    minCenterCluster = np.mat(np.zeros((k,n)))
    minClusterAssign = np.mat(np.zeros((m, 2)))

    for randCenterCount in range(maxRandCenterIter):
        #随机生成聚类中心
        centerCluster = generateRandCenterOfCluster(dataSet, k)

        # 记录点分配，第一列是样本所在簇，第二列是样本到中心距离
        clusterAssign = np.mat(np.zeros((m, 2)))

        #标记聚类中心是否还在改变
        isChanged = True

        iterCount = 0
        while isChanged and iterCount < maxIter:
            iterCount += 1
            isChanged = False

            for i in range(m):
                #计算第i个样本到聚类中心的距离
                minIndex = 0
                minDistance = np.inf
                for j in range(k):
                    distance = calEclud(dataSet[i,:], centerCluster[j,:])
                    if distance < minDistance:
                        minIndex = j
                        minDistance = distance
                if clusterAssign[i, 0] != minIndex:
                    isChanged = True
                #即使被分配的簇一样，距离也可能不一样
                clusterAssign[i,:] = minIndex, minDistance**2
            #刷新聚类中心
            for cent in range(k):
                # 通过数组过滤获得簇中的点
                ptsInCluster = dataSet[np.nonzero(
                    clusterAssign[:, 0].A == cent)[0]]
                if ptsInCluster.shape[0] > 0:
                    # 计算均值并移动
                    centerCluster[cent, :] = np.mean(ptsInCluster, axis=0)
        costValue = cost(clusterAssign)
        print(costValue)
        if costValue < minCost:
            minCost = costValue
            minCenterCluster = centerCluster
            minClusterAssign = clusterAssign

    return minCenterCluster, minClusterAssign, minCost




#Test
# data = loadDataSet("data\ex7data1.mat")
# centerCluster = generateRandCenterOfCluster(data, 3)
# print(centerCluster)
# minX = np.min(data[:,4])
# print(minX)
# X1 = np.array([1,2])
# X2 = np.array([3,4])
#
# result = calEclud(X1,X2)
# print(result)
# X1 = np.array([[1,2],
#                [3,4],
#                [5,6]])
# print(X1.shape)
# zerosArray = np.mat(np.zeros((3,5)))
# print(zerosArray)
# x = np.random.rand(3,1)
# print(x)

if __name__ == "__main__":
    dataMat = np.mat(loadDataSet("data\ex7data2.mat"))
    centroids, clusterAssment , costValue= kMeans(dataMat, 3)
    print(costValue)
    clusterCount = np.shape(centroids)[0]
    m = np.shape(dataMat)[0]
    # 绘制散点图
    patterns = ['o', 'D', '^', 's']
    colors = ['b', 'g', 'y', 'black']
    fig = plt.figure()
    title = 'kmeans with k=4'
    ax = fig.add_subplot(111, title=title)
    for k in range(clusterCount):
        # 绘制聚类中心
        ax.scatter(centroids[k, 0], centroids[k, 1], color='r', marker='+', linewidth=20)
        for i in range(m):
            # 绘制属于该聚类中心的样本
            ptsInCluster = dataMat[np.nonzero(clusterAssment[:, 0].A==k)[0]]
            ax.scatter(ptsInCluster[:, 0].flatten().A[0], ptsInCluster[:, 1].flatten().A[0], marker=patterns[k], color=colors[k])
    plt.show()

