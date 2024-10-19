import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 定义模型函数
def model(t, a, b, T):
    return a * np.sin(2 * np.pi * t / T) + b

# 定义误差函数
def error(params, t, y):
    a, b, T = params
    return np.sum((y - model(t, a, b, T))**2)

# 示例数据
t = np.linspace(0, 10, 100)  # 时间点
y = np.sin(2 * np.pi * t / 5) + np.random.normal(0, 0.1, t.shape)  # 实际信号

# 初始参数
initial_params = [1, 0, 5]  # a, b, T

# 最小化误差
result = minimize(error, initial_params, args=(t, y))

# 拟合参数
a_fit, b_fit, T_fit = result.x

# 使用拟合参数构造脉图
t_fit = np.linspace(0, 10, 100)
y_fit = model(t_fit, a_fit, b_fit, T_fit)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(t, y, label='Original Signal')
plt.plot(t_fit, y_fit, label='Fitted Signal')
plt.legend()
plt.title('Fitting Result')

plt.subplot(1, 2, 2)
plt.plot(t_fit, y_fit, label='Fitted Signal')
plt.title('Fitted Pulse Diagram')
plt.show()
