# 导入操作系统库
import os
# 更改工作目录
os.chdir(r"D:\softwares\applied statistics\pythoncodelearning\chap1\sourcecode")
# 导入基础计算库
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入数据分析库
import pandas as pd
# 导入模型评估的工具
# 导入数据集获取工具
from sklearn.datasets import load_diabetes
# 导入标准化处理工具
from sklearn.preprocessing import StandardScaler
# 导入Lasso信息准则估计器
from sklearn.linear_model import LassoLarsIC
# 导入管道操作
from sklearn.pipeline import make_pipeline
# 导入时间库
import time
# 导入绘图库中的字体管理包
from matplotlib import font_manager
# 实现中文字符正常显示
font = font_manager.FontProperties(fname=r"C:\Windows\Fonts\SimKai.ttf")
# 使用seaborn风格绘图
plt.style.use("seaborn-v0_8")
# 导入数据集
X, y = load_diabetes(return_X_y=True, as_frame=True)
# 在原始数据集中加入一些随机特征，增加变量
np.random.seed(42)
# 特征数
n_random_features = 14
# 生成随机的X
X_random = pd.DataFrame(
    np.random.randn(X.shape[0], n_random_features),
    columns=[f"random_{i:02d}" for i in range(n_random_features)],
)
# 合并X
X = pd.concat([X, X_random], axis=1)
# 查看下数据
print(X[X.columns[::3]].head())
# 计时开始
start_time = time.time()
# 建立lassoIC模型，它的alpha惩罚系数是自动生成的，无法指定
lasso_lars_aic = make_pipeline(
    StandardScaler(), # 数据标准化
    LassoLarsIC(criterion="aic") # 使用aic准则
)
# 模型拟合
lasso_lars_aic.fit(X, y)
# 记录模型使用的alpha
alpha_aic = lasso_lars_aic[-1].alpha_
# 建立lassoIC模型，它的alpha惩罚系数是自动生成的，无法指定
lasso_lars_bic = make_pipeline(
    StandardScaler(), # 数据标准化
    LassoLarsIC(criterion="bic") # 使用aic准则
)
# 模型拟合
lasso_lars_bic.fit(X, y)
# 拟合时间
fit_time = time.time() - start_time
print("模型拟合的时间为：", fit_time, sep="\n")
# 记录模型使用的alpha
alpha_bic = lasso_lars_bic[-1].alpha_
# 将alpha和AIC，BIC存储起来
results = pd.DataFrame(
    {
        "alphas": lasso_lars_aic[-1].alphas_,
        "AIC criterion": lasso_lars_aic[-1].criterion_,
        "BIC criterion": lasso_lars_bic[-1].criterion_
    }
).set_index("alphas")

# 定义一个函数，选择出最小的AIC对应的alpha
def highlight_min(x):
    x_min = x.min()
    return ["font-weight: bold" if v == x_min else "" for v in x]
# 高亮标记
print(results.style.apply(highlight_min))
# 最后，我们可以绘制不同alpha值的AIC和BIC值。
# 图中的垂直线对应于为每个标准选择的alpha。所选择的α对应于AIC或BIC准则的最小值。
fig1, ax = plt.subplots(figsize=(6,6))
ax = results.plot(ax=ax)
# 画竖直线
ax.vlines(
    alpha_aic,
    results["AIC criterion"].min(),
    results["AIC criterion"].max(),
    label="alpha: AIC estimate",
    linestyles="--",
    color="tab:blue",
)
ax.vlines(
    alpha_bic,
    results["BIC criterion"].min(),
    results["BIC criterion"].max(),
    label="alpha: BIC estimate",
    linestyle="--",
    color="tab:orange",
)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("criterion")
ax.set_xscale("log")
# 展示图例
ax.legend()
ax.set_title(
    f"Information-criterion for model selection (training time {fit_time:.2f}s)"
)
plt.show()
fig1.savefig("../codeimage/code11.pdf")
