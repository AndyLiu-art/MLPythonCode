# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap7\sourcecode")
# 导入时间库
import time
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入ExpSineSquared核
from sklearn.gaussian_process.kernels import ExpSineSquared
# 导入核岭回归中的KernelRidge核
from sklearn.kernel_ridge import KernelRidge
# 导入高斯过程回归器
from sklearn.gaussian_process import GaussianProcessRegressor
# 导入White核函数
from sklearn.gaussian_process.kernels import WhiteKernel
# 导入loguniform分布
from scipy.stats import loguniform
# 导入随机CV搜索包
from sklearn.model_selection import RandomizedSearchCV
# 导入岭回归模型
from sklearn.linear_model import Ridge
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成数据
rng = np.random.RandomState(0)
data = np.linspace(0, 30, num=1000).reshape(-1, 1)
target = np.sin(data).ravel()
# 训练样本的下标
training_sample_indices = rng.choice(
    np.arange(0, 400), size=40, replace=False
)
# 训练数据
training_data = data[training_sample_indices]
# 有噪声的y
training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(
    len(training_sample_indices)
)
fig1, ax = plt.subplots(figsize=(6,6))
ax.plot(data, target, label="True signal", linewidth=2)
ax.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
ax.legend()
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title(
    "Illustration of the true generative process and \n"
    "noisy measurements available during training"
)
fig1.savefig("../codeimage/code5.pdf")
plt.show()
# 构造岭回归模型
ridge = Ridge()
# 模型拟合
ridge.fit(training_data, training_noisy_target)
fig2, ax = plt.subplots(figsize=(6,6))
ax.plot(data, target, label="True signal", linewidth=2)
ax.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
# 绘制ridge预测的结果
ax.plot(data, ridge.predict(data), label="Ridge regression")
ax.legend()
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title("Limitation of a linear model such as ridge")
fig2.savefig("../codeimage/code6.pdf")
plt.show()
# 构造KernelRidge核岭回归模型
kernel_ridge = KernelRidge(kernel=ExpSineSquared())
start_time = time.time()
# 模型拟合
kernel_ridge.fit(training_data, training_noisy_target)
print(
    f"Fitting KernelRidge with default kernel: {time.time() - start_time:.3f} seconds"
)
fig3, ax = plt.subplots(figsize=(6,6))
ax.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
ax.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
ax.plot(
    data,
    kernel_ridge.predict(data), # 预测
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
ax.legend(loc="lower right")
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title(
    "Kernel ridge regression with an exponential sine squared\n "
    "kernel using default hyperparameters"
)
fig3.savefig("../codeimage/code7.pdf")
plt.show()
# 分布
param_distributions = {
    "alpha": loguniform(1e0, 1e3),
    "kernel__length_scale": loguniform(1e-2, 1e2), # 长度与scale参数
    "kernel__periodicity": loguniform(1e0, 1e1), # 周期参数
}
# 随机参数搜索，构造核岭回归模型
kernel_ridge_tuned = RandomizedSearchCV(
    kernel_ridge,
    param_distributions=param_distributions,
    n_iter=500,
    random_state=0,
)
start_time = time.time()
# 模型拟合
kernel_ridge_tuned.fit(training_data, training_noisy_target)
print(f"Time for KernelRidge fitting: {time.time() - start_time:.3f} seconds")
start_time = time.time()
# 预测
predictions_kr = kernel_ridge_tuned.predict(data)
print(f"Time for KernelRidge predict: {time.time() - start_time:.3f} seconds")
fig4, ax = plt.subplots(figsize=(6,6))
ax.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
ax.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
ax.plot(
    data,
    predictions_kr,
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
ax.legend(loc="lower right")
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title(
    "Kernel ridge regression with an exponential sine squared\n "
    "kernel using tuned hyperparameters"
)
fig4.savefig("../codeimage/code8.pdf")
plt.show()
# 定义核函数
kernel = 1.0 * ExpSineSquared(
    1.0, 5.0, periodicity_bounds=(1e-2, 1e1)
) + WhiteKernel(
    1e-1
)
# 构造GPR模型
gaussian_process = GaussianProcessRegressor(kernel=kernel)
start_time = time.time()
# 模型拟合
gaussian_process.fit(training_data, training_noisy_target)
print(
    f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds"
)
start_time = time.time()
# 模型预测
mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
    data,
    return_std=True,
)
print(
    f"Time for GaussianProcessRegressor predict: {time.time() - start_time:.3f} seconds"
)
fig5, ax = plt.subplots(figsize=(6,6))
ax.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
ax.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
# 绘制核岭回归的预测值
ax.plot(
    data,
    predictions_kr,
    label="Kernel ridge",
    linewidth=2,
    linestyle="dashdot",
)
# 绘制高斯过程回归的预测值
ax.plot(
    data,
    mean_predictions_gpr,
    label="Gaussian process regressor",
    linewidth=2,
    linestyle="dotted",
)
# 绘制高斯过程回归的预测的标准差范围
ax.fill_between(
    data.ravel(),
    mean_predictions_gpr - std_predictions_gpr,
    mean_predictions_gpr + std_predictions_gpr,
    color="tab:green",
    alpha=0.2,
)
ax.legend(loc="lower right")
ax.set_xlabel("data")
ax.set_ylabel("target")
ax.set_title("Comparison between kernel ridge and gaussian process regressor")
fig5.savefig("../codeimage/code9.pdf")
plt.show()
