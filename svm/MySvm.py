import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def linearKernel():
    #线性核函数
    def calc(X, A):
        return X* A.T
    return calc

def rbfKernel(delta):
    #rbf核函数

    gamma = 1.0/(2 * delta**2)

    def calc(X, A):
        #相比于自己实现，使用性能更好的sklearn.metrics.pairwise.rbf_kernel
        return np.mat(rbf_kernel(X, A, gamma=gamma))
    return calc

def getSmo(X, y, C, tol, maxIter, kernel = linearKernel()):
    '''
    
    :param X: 训练集
    :param y: 标签集
    :param C: 正规化参数
    :param tol: 容错率
    :param maxIter: 最大迭代次数
    :param kernel: 使用的核函数
    :return: 
        trainSimple 简化版SMO
        train 完整版SMO
        predict 预测函数
    '''
    m, n = X.shape

    K = kernel(X, X)
    #存放预测误差的缓存
    ECache = np.zeros((m,2))

    def w(alphas, supportVectorsIndex, supportVectors):
        '''
        模型只与支持向量相关，权值根据公式，由alpha，y,支持向量算出
        :param alphas: 拉格朗日乘子
        :param supportVectorsIndex:支持向量下标 
        :param supportVectors: 支持向量
        :return: 
        '''
        return (np.multiply(alphas[supportVectorsIndex], y[supportVectorsIndex]).T * supportVectors).T

    def error(i, alphas, b):
        '''
        预测误差
        :param i: 
        :param alphas: 拉格朗日乘子
        :param b: 常数
        :return: 
        '''
        f = float(np.multiply(alphas,y).T * K[:, i])+b
        error = f - float(y[i])
#        print("error:" + str(type(error))+str(error))
        return error

    def updateE(i, alphas, b):
        #更新误差缓存
        ECache[i] = [1, error(i, alphas, b)]

    def predict(X, alphas, b ,supportVectorsIndex, supportVectors):
        '''
        计算预测结果
        :param X: 输入的训练集
        :param alphas: 拉格朗日乘子
        :param b: 常数
        :param supportVectorsIndex: 支持向量下标
        :param supportVectors: 支持向量
        :return: 
            pridicts 预测结果，有0，-1,1三种值
        '''
        Ks = kernel(supportVectors, X)
        predicts = (np.multiply(alphas[supportVectorsIndex], y[supportVectorsIndex]).T * Ks + b).T
        predicts = np.sign(predicts)
        return predicts

    def select(i, alphas, b):
        #alpha对选择
        Ei = error(i, alphas, b)
        # 选择违背KKT条件的，作为alpha2
        Ri = y[i] * Ei
        if (Ri < -tol and alphas[i] < C) or (Ri > tol and alphas[i] > 0):
            # 选择第二个参数,这里使用启发式方法
            # j, Ej = selectJ(i, Ei, alphas, b)
            j = selectJRand(i)
            Ej = error(j, alphas, b)
            # L,H保证alpha在o到c之间
            if y[i] != y[j]:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])
            else:
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            if L == H:
                return 0, alphas, b
            #计算最优修改量
            Kii = K[i, i]
            Kjj = K[j, j]
            Kij = K[i, j]
            eta = 2.0 * Kij - Kii - Kjj

            if eta >= 0:
                return 0, alphas, b
            iOld = alphas[i].copy()
            jOld = alphas[j].copy()
            alphas[j] = jOld - y[j] * (Ei - Ej) / eta
            if alphas[j] > H:
                alphas[j] = H
            elif alphas[j] < L:
                alphas[j] = L
            if abs(alphas[j] - jOld) < tol:
                alphas[j] = jOld
                return 0, alphas, b
            alphas[i] = iOld + y[i] * y[j] * (jOld - alphas[j])
            # 更新ECache
            updateE(i, alphas, b)
            updateE(j, alphas, b)
            # 更新常数b
            bINew = b - Ei - y[i] * (alphas[i] - iOld) * Kii - y[j] * (alphas[j] - jOld) * Kij
            bJNew = b - Ej - y[i] * (alphas[i] - iOld) * Kij - y[j] * (alphas[j] - jOld) * Kjj
            if alphas[i] > 0 and alphas[i] < C:
                b = bINew
            elif alphas[j] > 0 and alphas[j] < C:
                b = bJNew
            else:
                b = (bINew + bJNew) / 2
            return 1, alphas, b
        else:
            return 0, alphas, b

    def selectJRand(i):
        #随机选择j
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j



    def selectJ(i, Ei, alphas, b):
        #选择误差差值最大的
        maxJ = 0
        maxDist = 0
        Ej = 0
        ECache[i] = [1, Ei]
        validCaches = np.nonzero(ECache[:, 0])[0]
        if len(validCaches) > 1:
            for k in validCaches:
                if k == i: continue
                Ek = error(k, alphas, b)
                dist = np.abs(abs(Ei - Ek))
                if maxDist < dist:
                    Ej = Ek
                    maxJ = k
                    maxDist = dist
            return maxJ, Ej
        else:
            ### 随机选择
            j = selectJRand(i)
            Ej = error(j, alphas, b)
            return j, Ej

    def train():
        """
        完整版训练算法

        :return: 
            alphas 拉格朗日乘子
            w 权值
            b 常数
            supportVectorsIndex 支持向量的坐标集
            supportVectors 支持向量
            iterCount 迭代次数
        """
        numChanged = 0
        examineAll = True
        iterCount = 0
        alphas = np.mat(np.zeros((m, 1)))
        b = 0
        # 如果所有alpha都遵从 KKT 条件，则在整个训练集上迭代
        # 否则在处于边界内 (0, C) 的 alpha 中迭代
        while (numChanged > 0 or examineAll) and (iterCount < maxIter):
            numChanged = 0
            if examineAll:
                for i in range(m):
                    changed, alphas, b = select(i, alphas, b)
                    numChanged += changed
            else:
                nonBoundIds = np.nonzero((alphas.A > 0) * (alphas.A < C))[0]
                for i in nonBoundIds:
                    changed, alphas, b = select(i, alphas, b)
                    numChanged += changed
            iterCount += 1

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
        supportVectorsIndex = np.nonzero(alphas.A > 0)[0]
        supportVectors = np.mat(X[supportVectorsIndex])
        return alphas, w(alphas, supportVectorsIndex, supportVectors), b, supportVectorsIndex, supportVectors, iterCount

    def trainSimple():
        """
        简化版训练算法
        :return: 
            alphas 拉格朗日乘子
            w 权值
            b 常数
            supportVectorsIndex 支持向量的坐标集
            supportVectors 支持向量
            iterCount 迭代次数
        """
        iterCount = 0
        alphas = np.mat(np.zeros((m, 1)))
        b = 0
        while iterCount < maxIter:
            numChanged = 0
            for i in range(m):
                Ei = error(i, alphas, b)
                Ri = y[i] * Ei
                # 选择违背KKT条件的，作为alpha2
                if (Ri < -tol and alphas[i] < C) or (Ri > tol and alphas[i] > 0):
                    # 选择第二个参数
                    j = selectJRand(i)
                    Ej = error(j, alphas, b)
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    Kii = K[i, i]
                    Kjj = K[j, j]
                    Kij = K[i, j]
                    eta = 2.0 * Kij - Kii - Kjj
                    if eta >= 0:
                        continue
                    iOld = alphas[i].copy();
                    jOld = alphas[j].copy()
                    alphas[j] = jOld - y[j] * (Ei - Ej) / eta
                    if alphas[j] > H:
                        alphas[j] = H
                    elif alphas[j] < L:
                        alphas[j] = L
                    if abs(alphas[j] - jOld) < tol:
                        alphas[j] = jOld
                        continue
                    alphas[i] = iOld + y[i] * y[j] * (jOld - alphas[j])

                    bINew = b - Ei - y[i] * (alphas[i] - iOld) * Kii - y[j] * (alphas[j] - jOld) * Kij
                    bJNew = b - Ej - y[i] * (alphas[i] - iOld) * Kij - y[j] * (alphas[j] - jOld) * Kjj
                    if alphas[i] > 0 and alphas[i] < C:
                        b = bINew
                    elif alphas[j] > 0 and alphas[j] < C:
                        b = bJNew
                    else:
                        b = (bINew + bJNew) / 2.0
                    numChanged += 1
            if numChanged == 0:
                iterCount += 1
            else:
                iterCount = 0
        supportVectorsIndex = np.nonzero(alphas.A > 0)[0]
        supportVectors = np.mat(X[supportVectorsIndex])
        return alphas, w(alphas, supportVectorsIndex, supportVectors), b, supportVectorsIndex, supportVectors, iterCount

    return trainSimple, train, predict

