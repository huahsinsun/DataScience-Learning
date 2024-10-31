# """计算圆周率"""
import numpy as np
import matplotlib.pyplot as plt

def simulation_pi(n_points):
    x = np.random.uniform(0,1,n_points)
    y = np.random.uniform(0,1,n_points)
    distance = lambda x,y:np.sqrt((x-0.5)**2+(y-0.5)**2)<=0.5
    z = distance(x,y)
    add_temp = np.count_nonzero(z)
    pi = add_temp/n_points*4
    print(pi)

simulation_pi(2000000)