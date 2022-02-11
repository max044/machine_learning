from statistics import mode
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

x, y = make_regression(n_samples=100, n_features=1, noise=10)
plt.scatter(x, y)

# print(x.shape)
y = y.reshape(y.shape[0], 1)
# print(y.shape)

X = np.hstack((x, np.ones(x.shape)))

theta = np.random.randn(2, 1)
# print(theta)

def model(X, theta):
    return X.dot(theta)


# plt.plot(model(X, theta), c='r')
# plt.show()


def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)

def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):

    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
    return theta

theta_final = gradient_descent(X, y, theta, learning_rate=0.001, n_iterations=1000)