import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_y_hat(b0, b1, x):
    n = len(x)
    y_hat = np.zeros(n)
    for i in range(n):
        y_hat[i] = b0 + b1 * x[i]
    return y_hat


def sum_square_error(y, y_hat):
    error = 0
    n = len(y)
    for i in range(n):
        error += (y[i] - y_hat[i]) ** 2
    return error


def get_b1(x, y):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    first_sum = 0
    second_sum = 0
    n = len(x)
    for i in range(n):
        first_sum += x[i] * (y[i] - y_bar)
    for i in range(n):
        second_sum += x[i] * (x[i] - x_bar)
    return first_sum / second_sum


def get_b0(x, y, b1):
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    b0 = y_bar - b1 * x_bar
    return b0


data = pd.read_csv(
    r"E:\Machine Learning\my course\new\3- Implmenting Simple Linear Regression\Simple linear regression.csv")

x = data["SAT"]
y = data["GPA"]
plt.scatter(x, y)
b1 = get_b1(x, y)
b0 = get_b0(x, y, b1)
y_hat = get_y_hat(b0, b1, x)
plt.plot(x, y_hat)
plt.show()
total_error = sum_square_error(y, y_hat)
print(total_error)
x_new = [1400, 1800, 1950]
y_pred = get_y_hat(b0, b1, x_new)
print(y_pred)
