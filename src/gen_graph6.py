'''
The most important regularization technique 
for GBRT is shrinkage: the idea is basically 
to do slow learning by shrinking the predictions 
of each individual tree by some small scalar, the 
learning_rate. By doing so the model has to re-enforce 
concepts. A lower learning_rate requires a higher number 
of n_estimators to get to the same level of training 
error — so its trading runtime against accuracy.
'''

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from itertools import islice
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


est = GradientBoostingRegressor(n_estimators=1000,
                                max_depth=1,
                                learning_rate=1.0)
est.fit(X_train, y_train)
n_estimators = len(est.estimators_)


def deviance_plot(est,
                  X_test,
                  y_test,
                  ax=None,
                  label='',
                  train_color='#2c7bb6',
                  test_color='#d7191c',
                  alpha=1.0):
    """Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error. """
    test_dev = np.empty(n_estimators)
    for i, pred in enumerate(est.staged_predict(X_test)):
        test_dev[i] = est.loss_(y_test, pred)
    if ax is None:
        fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(np.arange(n_estimators) + 1,
            test_dev,
            color=test_color,
            label='Test %s' % label,
            linewidth=2,
            alpha=alpha)
    ax.plot(np.arange(n_estimators) + 1,
            est.train_score_,
            color=train_color,
            label='Train %s' % label,
            linewidth=2,
            alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 2))
    return test_dev, ax


def fmt_params(params):
    return ", ".join("{0}={1}".format(key, val) for key, val in params.items())


fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
for params, (test_color, train_color) in [({}, ('#d7191c', '#2c7bb6')),
                                          ({
                                              'learning_rate': 0.1
                                          }, ('#fdae61', '#abd9e9'))]:
    est = GradientBoostingRegressor(n_estimators=n_estimators,
                                    max_depth=1,
                                    learning_rate=1.0)
    est.set_params(**params)
    est.fit(X_train, y_train)
    test_dev, ax = deviance_plot(est,
                                 X_test,
                                 y_test,
                                 ax=ax,
                                 label=fmt_params(params),
                                 train_color=train_color,
                                 test_color=test_color)

    ax.annotate(
        'Requires more trees',
        xy=(200, est.train_score_[199]),
        xycoords='data',
        xytext=(300, 1.0),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
    )
ax.annotate(
    'Lower test error',
    xy=(900, test_dev[899]),
    xycoords='data',
    xytext=(600, 0.5),
    textcoords='data',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
)
plt.legend(loc='upper right')

plt.show()