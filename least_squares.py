# filename: main.py
#
# by: Abhay Gupta
# date: 4/14/19
#
# desc: computes least square regression on dataset
# source: https://www.cs.auckland.ac.nz/courses/compsci369s1c/lectures/GG-notes/CS369-LeastSquares.pdf
###

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Create an array
x = np.arange(10)
y = np.random.randint(low=1, high=10, size=10)

# Preprocessing Input data
data = pd.read_csv('data.csv')
x = data.iloc[:, 0]
y = data.iloc[:, 1]

n = len(x)

u1 =  np.ones(n)
A = np.concatenate([u1, x], axis = 0)
A = np.reshape(A, [n, 2], order = 'F')

B = np.matmul(A.T, A)
B = np.linalg.inv(B)

A = np.matmul(B, A.T)
u = np.matmul(A, y)

y_est = u[0] + u[1]*x

plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_est), max(y_est)], color = 'red')
plt.show()
