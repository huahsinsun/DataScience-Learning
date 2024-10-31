"""随机抽样是蒙特卡罗方法的核心技术，让我们来了解几种基本的抽样方法！"""
import numpy as np

# uniform Sampling 均匀抽样

uniform_series = np.random.uniform(0,10,10) # 生成[0-10)之间的随机数10个
print(uniform_series) # 在给定范围内，每个点被选中的概率都相等

# Direct Sampling 直接采样 当我们知道概率的分布形式时
# 正态分布
normal_samples = np.random.normal(0,1,10)
print(normal_samples)