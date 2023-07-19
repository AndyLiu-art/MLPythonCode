# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap10\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入MLP分类器
from sklearn.neural_network import MLPRegressor
# 导入数据集工具
from sklearn.datasets import load_diabetes
# 导入数据集划分工具
from sklearn.model_selection import train_test_split
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 加载数据
X, y = load_diabetes(return_X_y=True)
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3
)
# 构建MLP回归模型
mlpr = MLPRegressor()
# 模型拟合
mlpr.fit(x_train, y_train)
# 预测
y_pred = mlpr.predict(x_test)
# MSE
mse = np.mean((y_pred-y_test)**2)
print("测试集上的MSE为", mse, sep="\n")
# 绘图
fig, ax = plt.subplots(figsize=(6,6))
ax.plot(y_test, y_pred, "ro", alpha=0.4)
plt.show()
fig.savefig("../codeimage/code3.pdf")
