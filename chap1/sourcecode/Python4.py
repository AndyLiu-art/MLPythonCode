# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap1\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入线性回归模型
from sklearn.linear_model import RidgeClassifier
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入混淆矩阵
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 维度
p = 200
# 样本量
samplesize = 100
# 真实的系数值
coef = np.random.normal(size=p)
# 生成X数据
x = np.random.randn(samplesize, p)
# 生成概率
p = np.exp(x.dot(coef))/(1+np.exp(x.dot(coef)))
# 生成y
y = np.random.binomial(1, p)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=10
)
# 构造Ridge分类器
clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
# 模型拟合
clf.fit(x_train, y_train)
# 预测
y_pred = clf.predict(x_test)
# 混淆矩阵
res = confusion_matrix(y_test, y_pred, labels=[0,1])
print("混淆矩阵为：", res, sep="\n")
# 开始绘图
fig, ax = plt.subplots(figsize=(6, 6))
# 绘制混淆矩阵图
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
# 设置横纵轴刻度
ax.xaxis.set_ticklabels([0,1])
ax.yaxis.set_ticklabels([0,1])
plt.show()
fig.savefig("../codeimage/code4.pdf")
