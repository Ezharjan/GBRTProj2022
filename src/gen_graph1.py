'''
A regression problem with one feature x and
the corresponding response y. 
We draw 100 training data points by picking 
an x coordinate uniformly at random, 
evaluating the ground truth (sinoid function; 
light blue line) and then adding some random 
gaussian noise. In addition to the 100 
training points (blue) we also draw 100 test 
data points (red) which we will use the evaluate our approximation.
'''

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
    plt.show()


plot_data(figsize=(8, 5))
