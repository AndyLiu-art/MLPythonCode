# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap8\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入数据划分工具
from sklearn.model_selection import train_test_split
# 导入Gassian朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB
# 导入数据获取工具
from sklearn.datasets import load_iris
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 获取数据
X, y = load_iris(return_X_y=True)
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)
# 构建模型
gnb = GaussianNB()
# 模型拟合
gnb.fit(X_train, y_train)
# 预测
y_pred = gnb.predict(X_test)
print(y_pred[1:5])
# 分类准确率
corrate = np.mean(y_pred==y_test)
print("分类准确率为：")
print(corrate)
