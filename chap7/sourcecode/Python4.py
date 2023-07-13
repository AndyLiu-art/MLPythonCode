# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap7\sourcecode")
# 导入基础计算库
import numpy as np
# 导入数据分析库
import pandas as pd
# 导入绘图库
import matplotlib.pyplot as plt
# 导入ExpSineSquared核
from sklearn.gaussian_process.kernels import RBF
# 导入高斯过程回归器
from sklearn.gaussian_process import GaussianProcessClassifier
# 导入数据集工具
from sklearn import datasets
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成数据
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)
# meshgrid中的步长
h = 0.02
# 定义核函数
kernel = 1.0 * RBF([1.0])
# 构造GPC模型
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel)
# 模型拟合
gpc_rbf_isotropic.fit(X, y)
# 构造核函数
kernel = 1.0 * RBF([1.0, 1.0])
# 构造GPC模型
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel)
# 模型拟合
gpc_rbf_anisotropic.fit(X, y)
# meshgrid数据
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

titles = ["Isotropic RBF", "Anisotropic RBF"]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
    ax = axs[i]
    # 预测概率
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    ax.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")
    ax.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    ax.set_xlabel("Sepal length")
    ax.set_ylabel("Sepal width")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(
        "%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta))
    )
plt.tight_layout()
fig.savefig("../codeimage/code13.pdf")
plt.show()
