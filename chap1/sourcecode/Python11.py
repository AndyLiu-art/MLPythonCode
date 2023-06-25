# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap1\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入Lasso模型
from sklearn.linear_model import MultiTaskLasso, Lasso
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 设置样本量，维度，回归模型中y的维度（响应变量的多元回归）
n_samples, n_features, n_tasks = 100, 30, 40
# 显著变量的个数
n_relevant_features = 5
# 初始化真实系数，是一个矩阵
coef = np.zeros((n_tasks, n_features))
# 时刻
times = np.linspace(0, 2 * np.pi, n_tasks)
# 设置随机数种子
np.random.seed(10)
# 生成真实系数
for k in range(n_relevant_features):
    coef[:, k] = np.sin(
        (1.0 + np.random.randn(1)) * times + 3 * np.random.randn(1)
    )
# 生成X
X = np.random.randn(n_samples, n_features)
# 生成Y
Y = np.dot(X, coef.T) + np.random.randn(n_samples, n_tasks)
print("查看多元响应变量Y的情况：", Y[:5, :2], sep="\n")
# 建立Lasso模型，分别对Y的每一个分量做，提取系数
coef_lasso_ = np.array(
    [
        Lasso(alpha=0.5).fit(X, y).coef_ for y in Y.T
    ]
)
# 建立MultiLasso模型，提取系数
coef_multi_task_lasso_ = MultiTaskLasso(alpha=1.0).fit(X, Y).coef_
# 开始绘图
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
# 用于展示稀疏二维数组的图形
axs[0].spy(coef_lasso_)
axs[0].set_xlabel("Feature")
axs[0].set_ylabel("Time (or Task)")
axs[0].text(10, 5, "Lasso")
axs[1].spy(coef_multi_task_lasso_)
axs[1].set_xlabel("Feature")
axs[1].set_ylabel("Time (or Task)")
axs[1].text(10, 5, "MultiTaskLasso")
fig.suptitle("Coefficient non-zero location")
plt.show()
fig.savefig("../codeimage/code15.pdf")
# 绘制第一个特征前的系数
feature_to_plot = 0
# 开始绘图
fig1, ax = plt.subplots(figsize=(6,6), tight_layout=True)
# 绘制coef的线图
ax.plot(
    coef[:, feature_to_plot], 
    color="seagreen", 
    linewidth=2, 
    label="Ground truth"
)
# 绘制coef_lass0的线图
ax.plot(
    coef_lasso_[:, feature_to_plot], 
    color="cornflowerblue", 
    linewidth=2, 
    label="Lasso"
)
# 绘制coef_task_lasso的线图
ax.plot(
    coef_multi_task_lasso_[:, feature_to_plot],
    color="gold",
    linewidth=2,
    label="MultiTaskLasso",
)
# 显示图例
ax.legend(loc="best")
# 设置纵轴范围
ax.set_ylim([-1.1, 1.1])
plt.show()
fig1.savefig("../codeimage/code16.pdf")
