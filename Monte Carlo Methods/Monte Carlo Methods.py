"""计算圆周率"""
import numpy as np
import matplotlib.pyplot as plt
from mpmath import sqrtm
import math


def estimate_pi(n_points):
    x = np.random.uniform(0,1,n_points)
    y = np.random.uniform(0,1,n_points)
    scaler_square = 1
    points = zip(x,y)
    add_temp = 0
    for point in points:
        if math.sqrt((point[0]-0.5)**2+(point[1]-0.5)**2)<=0.5:
            add_temp += 1
    scaler = add_temp/n_points
    print(scaler*scaler_square)



estimate_pi(20000)
