import numpy as np
import sys

def normalize(X):
    '''
    对数据标准化处理
    :param X: 数据集
    :return: 
        xNormalize 标准化之后的数据
    '''
    xNormalize = X.copy()

    xMeans = np.mean(xNormalize, axis=0)
    xStd = np.std(xNormalize, axis=0)
    xNormalize = (xNormalize - xMeans)/xStd
    return xNormalize

def pca(X, k=1):
    '''
    降维
    :param X:数据集 
    :param k: 降到k维
    :return: 
        
    '''
    #数据归一化
    xNorm = normalize(X)
    #计算协方差矩阵
    covMat = np.cov(xNorm, rowvar=0)
    #奇异值分解
    U, S, V = np.linalg.svd(covMat)
    #取出前k个向量
    uReduce = U[:, 0:k]
    xReduce = np.dot(xNorm, uReduce)
    return xNorm, xReduce, U, uReduce, S, V

def recoverData(uReduce, xReduce):
    return np.dot(xReduce, uReduce.T)

# if __name__ == '__main__':
#
#     x = np.array([[7,8,9],
#                  [4,5,6],
#                   [1,2,3]
#                   ])
#     w, v = np.linalg.eig(x)
#     print(w)
#     print(v)
#
#     U, S, V = np.linalg.svd(x)
#     print(U)
#     print(S)
#     print(V)
# print(sys.path)