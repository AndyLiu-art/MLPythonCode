# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap5\sourcecode")
# 导入绘图库
import matplotlib.pyplot as plt
# 导入数据生成工具
from sklearn.datasets import load_iris
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier
# 导入绘制树状图的工具
from sklearn.tree import plot_tree
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 获取数据
iris = load_iris()
# 构建模型
clf = DecisionTreeClassifier()
# 模型拟合
clf.fit(iris.data, iris.target)
fig, ax = plt.subplots(figsize=(6,6), tight_layout=True)
plot_tree(clf, filled=True)
ax.set_title("Decision tree trained on all the iris features")
plt.show()
fig.savefig("../codeimage/code2.pdf")
