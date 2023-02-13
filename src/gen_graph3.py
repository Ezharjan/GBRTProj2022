'''
Now, let’s fit a gradient boosting model to the 
training data and let’s see how the approximation 
progresses as we add more and more trees. The scikit-learn 
gradient boosting estimators allow us to evaluate the 
prediction of a model as a function of the number of trees 
via the staged_(predict|predict_proba) methods. These 
return a generator that iterates over the predictions as 
we continue to add more and more trees.
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


plot_data()

est = GradientBoostingRegressor(n_estimators=1000,
                                max_depth=1,
                                learning_rate=1.0)
est.fit(X_train, y_train)
ax = plt.gca()
first = True
# step over prediction as we added 20 more trees.
for pred in islice(est.staged_predict(x_plot[:, np.newaxis]), 0, 1000, 10):
    plt.plot(x_plot, pred, color='r', alpha=0.2)
if first:
    ax.annotate('High bias - low variance',
                xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
                xycoords='data',
                xytext=(3, 4),
                textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
first = False
pred = est.predict(x_plot[:, np.newaxis])
plt.plot(x_plot, pred, color='r', label='GBRT max_depth=1')
ax.annotate('Low bias - high variance',
            xy=(x_plot[x_plot.shape[0] // 2], pred[x_plot.shape[0] // 2]),
            xycoords='data',
            xytext=(6.25, -6),
            textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
plt.legend(loc='upper left')

plt.show()
'''
The above plot shows 50 red lines where each shows 
the response of the GBRT model after 20 trees have 
been added. It starts with a very crude approximation 
that can only fit more-or-less constant functions (i.e. 
High bias – low variance) but as we add more trees the 
more variance our model can capture resulting in the 
solid red line.
We can see that the more trees we add to our GBRT model 
and the deeper the individual trees are the more variance 
we can capture thus the higher the complexity of our model. 
But as usual in machine learning model complexity comes at 
a price — overfitting.
An important diagnostic when using GBRT in practice is the 
so-called deviance plot that shows the training/testing error 
(or deviance) as a function of the number of trees.
'''