'''
We fit an individual regression tree to the 
above data and acquire a piece-wise constant 
approximation. The deeper we grow the tree, 
the more constant segments we can accommodate 
and thus, the more variance we can capture.
'''

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np


def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)


def gen_data(n_samples=200):
    """generate training and testing data"""
    np.random.seed(13)
    x = np.random.uniform(0, 10, size=n_samples)
    x.sort()
    y = ground_truth(x) + 0.75 * np.random.normal(size=n_samples)
    train_mask = np.random.randint(0, 2, size=n_samples).astype(bool)
    x_train, y_train = x[train_mask, np.newaxis], y[train_mask]
    x_test, y_test = x[~train_mask, np.newaxis], y[~train_mask]
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = gen_data(200)

# plot ground truth
x_plot = np.linspace(0, 10, 500)


def plot_data(figsize=(8, 5)):
    fig = plt.figure(figsize=figsize)
    gt = plt.plot(x_plot,
                  ground_truth(x_plot),
                  alpha=0.4,
                  label='ground truth')
    # plot training and testing data
    plt.scatter(X_train, y_train, s=10, alpha=0.4)
    plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
    plt.xlim((0, 10))
    plt.ylabel('y')
    plt.xlabel('x')
    # plt.show()

plot_data()

est = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
plt.plot(x_plot,
         est.predict(x_plot[:, np.newaxis]),
         label='RT max_depth=1',
         color='g',
         alpha=0.9,
         linewidth=2)
est = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
plt.plot(x_plot,
         est.predict(x_plot[:, np.newaxis]),
         label='RT max_depth=3',
         color='g',
         alpha=0.7,
         linewidth=1)
plt.legend(loc='upper left')
plt.show()