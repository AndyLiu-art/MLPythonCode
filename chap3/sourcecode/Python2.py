# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap3\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入支持向量机模型
from sklearn import svm
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成样本
xx, yy = np.meshgrid(
    np.linspace(-3, 3, 500), 
    np.linspace(-3, 3, 500)
)
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
# 拟合NuSVM模型
clf = svm.NuSVC(gamma="auto")
# 模型拟合
clf.fit(X, Y)
# 决策函数
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# 开始绘图
fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
# 图形
ax.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
)
# 等高线
contours = ax.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
ax.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired, edgecolors="k")
ax.set_xticks(())
ax.set_yticks(())
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
plt.show()
fig.savefig("../codeimage/code2.pdf")
