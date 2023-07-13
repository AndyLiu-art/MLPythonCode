# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap7\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入高斯过程回归器
from sklearn.gaussian_process import GaussianProcessRegressor
# 导入径向核以及White核
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成数据
def target_generator(X, add_noise=False):
    target = 0.5 + np.sin(3 * X)
    if add_noise:
        rng = np.random.RandomState(1)
        target += rng.normal(0, 0.3, size=target.shape)
    return target.squeeze()
# 生成数据X
X = np.linspace(0, 5, num=30).reshape(-1, 1)
# 生成数据y，没有噪声
y = target_generator(X, add_noise=False)
# 可视化生成的无噪声数据
fig1, ax = plt.subplots(figsize=(6,6))
ax.plot(X, y, label="Expected signal")
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
fig1.savefig("../codeimage/code1.pdf")
plt.show()
# 生成有噪声的数据
rng = np.random.RandomState(0)
# 生成数据X
X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
# 生成数据y
y_train = target_generator(X_train, add_noise=True)
# 继续绘图
fig2, ax = plt.subplots(figsize=(6,6))
ax.plot(X, y, label="Expected signal")
ax.scatter(
    x=X_train[:, 0],
    y=y_train,
    color="black",
    alpha=0.4,
    label="Observations",
)
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
fig2.savefig("../codeimage/code2.pdf")
plt.show()
# 定义核函数
kernel = 1.0 * RBF(
    length_scale=1e1, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(
    noise_level=1, noise_level_bounds=(1e-5, 1e1)
)
# 构造高斯回归器
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
# 模型拟合
gpr.fit(X_train, y_train)
# 模型预测
y_mean, y_std = gpr.predict(X, return_std=True)
fig3, ax = plt.subplots(figsize=(6,6))
ax.plot(X, y, label="Expected signal")
ax.scatter(
    x=X_train[:, 0], 
    y=y_train, 
    color="black", 
    alpha=0.4, 
    label="Observations"
)
# 绘制误差棒，均值加标准差
ax.errorbar(X, y_mean, y_std)
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title(
    (
        f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
        f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
    ),
    fontsize=8,
)
fig3.savefig("../codeimage/code3.pdf")
plt.show()
# 构造另一个核函数
kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
    noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)
)
# 模型构造
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
# 模型拟合
gpr.fit(X_train, y_train)
# 模型预测
y_mean, y_std = gpr.predict(X, return_std=True)
fig4, ax = plt.subplots(figsize=(6,6))
ax.plot(X, y, label="Expected signal")
ax.scatter(
    x=X_train[:, 0], 
    y=y_train, 
    color="black", 
    alpha=0.4, 
    label="Observations"
)
ax.errorbar(X, y_mean, y_std)
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title(
    (
        f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
        f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}"
    ),
    fontsize=8,
)
fig4.savefig("../codeimage/code4.pdf")
plt.show()