# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap3\sourcecode")
# 导入绘图库
import matplotlib.pyplot as plt
# 导入支持向量机模型
from sklearn import svm
# 导入数据生成工具
from sklearn.datasets import make_blobs
# 导入决策边界显示工具
from sklearn.inspection import DecisionBoundaryDisplay
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成样本
X, y = make_blobs(
    n_samples=40, # 样本量
    centers=2, # 两个类
    random_state=6
)
# 建立没有正则化的SVM
clf = svm.SVC(kernel="linear", C=1000)
# 模型拟合
clf.fit(X, y)
# 绘图
fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# 绘制决策边界
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax
)
# 绘制支持向量（在支持向量上的点）
ax.scatter(
    clf.support_vectors_[:, 0], # 支持向量
    clf.support_vectors_[:, 1], # 支持向量
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()
fig.savefig("../codeimage/code1.pdf")
