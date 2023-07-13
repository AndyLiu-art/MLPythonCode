# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap7\sourcecode")
# 导入日期库
import datetime
# 导入基础计算库
import numpy as np
# 导入数据分析库
import pandas as pd
# 导入绘图库
import matplotlib.pyplot as plt
# 导入ExpSineSquared核
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF
# 导入高斯过程回归器
from sklearn.gaussian_process import GaussianProcessRegressor
# 导入White核函数
from sklearn.gaussian_process.kernels import WhiteKernel, RationalQuadratic
# 导入loguniform分布
from scipy.stats import loguniform
# 导入数据集工具
from sklearn.datasets import fetch_openml
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 获取数据
co2 = fetch_openml(data_id=41187, as_frame=True, parser="pandas")
print(co2.frame.head())
# 提取CO2和日期两列
co2_data = co2.frame
co2_data["date"] = pd.to_datetime(co2_data[["year", "month", "day"]])
co2_data = co2_data[["date", "co2"]].set_index("date")
print(co2_data.head())
# 绘制原始数据的时间序列线图
fig1, ax = plt.subplots(figsize=(6,6))
co2_data.plot(ax=ax)
ax.set_ylabel("CO$_2$ concentration (ppm)")
ax.set_title("Raw air samples measurements from the Mauna Loa Observatory")
fig1.savefig("../codeimage/code10.pdf")
plt.show()
# 绘制月份平均值的图
fig2, ax = plt.subplots(figsize=(6,6))
# 计算月平均数据
co2_data = co2_data.resample("M").mean().dropna(axis="index", how="any")
co2_data.plot(ax=ax)
ax.set_ylabel("Monthly average of CO$_2$ concentration (ppm)")
ax.set_title(
    "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
)
fig2.savefig("../codeimage/code11.pdf")
plt.show()
# 确定X和Y
X = (co2_data.index.year + co2_data.index.month / 12).to_numpy().reshape(-1, 1)
y = co2_data["co2"].to_numpy()
print(X[:3])
print(y[:3])
# 长期趋势核
long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
# 构造季节核函数
seasonal_kernel = (
    2.0**2
    * RBF(length_scale=100.0)
    * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
)
# 构造不规则因素的核
irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
# 构造噪声核
noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
)
# 最终的核函数
co2_kernel = (
    long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
)
# 开始建模
y_mean = y.mean()
# 构造GPR模型
gaussian_process = GaussianProcessRegressor(kernel=co2_kernel, normalize_y=False)
# 模型拟合
gaussian_process.fit(X, y - y_mean)
# 记录事件
today = datetime.datetime.now()
current_month = today.year + today.month / 12
X_test = np.linspace(start=1958, stop=current_month, num=1_000).reshape(-1, 1)
# 预测
mean_y_pred, std_y_pred = gaussian_process.predict(X_test, return_std=True)
mean_y_pred += y_mean
# 可视化结果
fig3, ax = plt.subplots(figsize=(6,6))
ax.plot(X, y, color="black", linestyle="dashed", label="Measurements")
ax.plot(X_test, mean_y_pred, color="tab:blue", alpha=0.4, label="Gaussian process")
ax.fill_between(
    X_test.ravel(),
    mean_y_pred - std_y_pred,
    mean_y_pred + std_y_pred,
    color="tab:blue",
    alpha=0.2,
)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("Monthly average of CO$_2$ concentration (ppm)")
ax.set_title(
    "Monthly average of air samples measurements\nfrom the Mauna Loa Observatory"
)
fig3.savefig("../codeimage/code12.pdf")
plt.show()
