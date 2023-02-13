'''
The blue line in the generated image shows the training error: 
it rapidly decreases in the beginning and then gradually slows 
down but keeps decreasing as we add more and more trees. The 
testing error (red line) too decreases rapidly in the beginning 
but then slows down and reaches its minimum fairly early 
(~50 trees) and then even starts increasing. This is what 
we call overfitting, at a certain point the model has so much 
capacity that it starts fitting the idiosyncrasies of the training 
data — in our case the random gaussian noise component that we 
added — and hence limiting its ability to generalize to new unseen 
data. A large gap between training and testing error is usually a 
sign of overfitting.
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


# plot_data()

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


test_dev, ax = deviance_plot(est, X_test, y_test)
ax.legend(loc='upper right')
# add some annotations
ax.annotate(
    'Lowest test error',
    xy=(test_dev.argmin() + 1, test_dev.min() + 0.02),
    xycoords='data',
    xytext=(150, 1.0),
    textcoords='data',
    arrowprops=dict(arrowstyle="->", connectionstyle="arc"),
)

ax.text(810, 0.25, 'train-test gap')

plt.show()