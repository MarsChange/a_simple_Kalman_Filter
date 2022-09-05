import numpy as np
import matplotlib.pyplot as plt
import random
# 定义样本数为100
samples_num = 100

# 定义基准值为1
standard = 1

# 定义初始状态的X
X = np.matrix([[1], [0]])

# 定义初始状态的协方差矩阵P
P = np.eye(2)

# 定义状态转移矩阵F（或者A）,F[1][3]和F[2][4]为时间t，假设的是每分钟采样一次，故值定为1
t = 1
F = np.matrix([[1, t],
               [0, 1]])

# 定义观测矩阵H
H = np.matrix([1, 0])

# 定义过程激励噪声协方差Q
Q = np.matrix([[0.001, 0],
               [0, 0.001]])

# 定义观测噪声协方差R
R = np.matrix([1000])

def drawBasicLine():
    # 先画出水平运动的一条标准直线
    x = np.linspace(standard, 100, samples_num)
    y = 1 + 0 * x
    plt.plot(x, y, label='origin_line')
    return x, y

# 利用random库添加符合高斯分布的噪声
def addGaussianNoise(x, y):
    # sigma为高斯分布的方差
    sigma = 0.04
    for i in range(len(x)):
        # 0为高斯分布的均值
        while True:
            noise = random.gauss(0, sigma)
            # 控制噪声在基准值的1%到10%之间
            if standard * 0.01 <= abs(noise) <= standard * 0.1:
                y[i] += noise
                break
    plt.plot(x, y, label='add_Gaussian_noise')

def KalmanFiltering(f_mat, p_mat, q_mat, x_mat, h_mat, r_mat, x, y):
    # 进行预测
    y_list = []
    # 初始化均方误差MSE为0
    mse = 0
    for i in range(samples_num):
        y_predict = np.dot(f_mat, x_mat)
        p_predict = np.dot(np.dot(f_mat, p_mat), f_mat.T) + q_mat
        k = np.dot(p_predict, h_mat.T) / (np.dot(np.dot(h_mat, p_predict), h_mat.T) + r_mat)
        x_mat = y_predict + np.dot(k, (y[i] - np.dot(h_mat, y_predict)))
        p_mat = p_predict - np.dot(np.dot(k, h_mat), p_predict)
        y_list.append(x_mat[0, 0])
    plt.plot(x, y_list)
    # 计算MSE
    for i in range(samples_num):
        mse += pow(y_list[i] - y[i], 2)/samples_num
    print('MSE = {}'.format(mse))


if __name__ == '__main__':
    x_, y_ = drawBasicLine()
    addGaussianNoise(x_, y_)
    KalmanFiltering(F, P, Q, X, H, R, x_, y_)
    plt.savefig('./run.png')
    plt.show()