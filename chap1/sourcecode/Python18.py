# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap1\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入logistic回归模型
from sklearn.linear_model import LogisticRegression
# 导入数据集
from sklearn.datasets import load_iris
# 导入标准化工具
from sklearn.preprocessing import StandardScaler
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 生成数据集
X, y = load_iris(return_X_y=True, as_frame=True)
# 选择y的两个标签对应的数据
X = X[y != 2]
y = y[y != 2]
# 对X标准化
X = StandardScaler().fit_transform(X)
# 建立带惩罚的Logistic模型
clf1 = LogisticRegression(
    penalty="l1",
    solver="liblinear"
)
# 建立不带惩罚的Logistic模型
clf2 = LogisticRegression(
    penalty=None
)
# 模型拟合
clf1.fit(X, y)
clf2.fit(X, y)
# 输出系数
print(clf1.coef_)
print(clf2.coef_)
# 标签预测
y1_pred = clf1.predict(X)
y2_pred = clf2.predict(X)
# 概率预测
prob1_pred = clf1.predict_proba(X)
prob2_pred = clf2.predict_proba(X)
# 输出标签
print(y1_pred)
print(y2_pred)
# 输出概率
print(prob1_pred[:10,])
print(prob2_pred[:10,])
