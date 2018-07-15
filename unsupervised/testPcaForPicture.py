import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import Pca as pca


def display(images, width, height):
    '''
    展示图片
    :param images:图片 
    :param width: 图片宽
    :param height: 高
    :return: 
    '''
    m, n = images.shape
    rows = int(np.floor(np.sqrt(m)))
    cols = int(np.ceil(m / rows))
    # 图像拼接
    dstImage = np.zeros((rows * height, cols * width))
    for i in range(rows):
        for j in range(cols):
            idx = cols * i + j
            image = images[idx].reshape(height, width)
            dstImage[i * height:i * height + height,
                     j * width: j * width + width] = image
    plt.imshow(dstImage.T, cmap='gray')
    plt.axis('off')
    plt.show()

data = loadmat('data/ex7faces.mat')
X = np.mat(data['X'],dtype=np.float32)
m, n = X.shape

# 展示原图
display(X[0:100, :], 32, 32)

XNorm, Z, U, UReduce, S, V = pca.pca(X, k=100)
XRec = pca.recoverData(UReduce, Z)

# 显示修复后的图，可以看出，PCA 损失了一部分细节
display(XRec[0:100, :], 32, 32)


