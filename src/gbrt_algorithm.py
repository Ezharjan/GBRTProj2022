from trees_data_structures import *
import numpy as np
import pandas as pd
import random


def get_optimal_partition(data, label_name):
    x = data.drop(label_name, axis=1)
    y = data[label_name]

    min_so_far = np.inf
    best_col_name, best_split_value = None, None

    for col_name in x.columns:
        col_unique_values = x[col_name].unique()
        for unique_value in col_unique_values:
            y_left = y[x[col_name] <= unique_value]
            y_right = y[x[col_name] > unique_value]

            loss = sum((y_left - y_left.mean())**2) + sum((y_right - y_right.mean())**2)

            if loss < min_so_far:
                min_so_far = loss
                best_col_name = col_name
                best_split_value = unique_value
    return best_col_name, best_split_value


def cart(data, max_depth, min_node_size, label_name):
    tree = RegressionTree()

    tree_levels_list = {0: [(data, tree.get_root())]}

    for depth in range(max_depth):
        tree_levels_list[depth+1] = []
        for node_data, node_reference in tree_levels_list[depth]:  # for each depth, iterate over all nodes and split as necessary
            col_name, split_value = get_optimal_partition(node_data, label_name)
            left_node_data = node_data[node_data[col_name] <= split_value]
            right_node_data = node_data[node_data[col_name] > split_value]

            # checks minimum node size violation
            if len(left_node_data.index) > min_node_size and len(right_node_data.index) > min_node_size:
                # define split parameters
                node_reference.split(col_name, split_value)

                # append descendants to the next depth
                tree_levels_list[depth+1].append((left_node_data, node_reference.left_descendant))
                tree_levels_list[depth+1].append((right_node_data, node_reference.right_descendant))
            else:
                node_reference.set_const(node_data[label_name])

    # set all nodes in max depth to nodes
    for node_data, node_reference in tree_levels_list[max_depth]:
        node_reference.set_const(node_data[label_name])

    return tree


def gbrt(train_data, test_data, label_name, params):
    tree_ensemble = RegressionTreeEnsemble()

    y_train = train_data[label_name].copy()
    y_test = test_data[label_name]

    f = pd.Series(data=np.zeros_like(y_train), index=y_train.index)
    for m in range(params.num_trees):
        grad = y_train - f
        train_data[label_name] = grad
        sub_data = train_data.sample(frac=params.sub_samp)
        tree = cart(sub_data, params.max_depth, params.min_node_size, label_name)

        # tree.root.print_sub_tree()

        y_tree_pred = train_data.apply(lambda xi: tree.evaluate(xi[:]), axis=1)
        weight = sum(grad * y_tree_pred) / sum(y_tree_pred ** 2)
        tree_ensemble.add_tree(tree, weight)
        f += params.weight_decay * weight * y_tree_pred

        # evaluate train and test sets
        y_train_ensemble_pred = train_data.apply(lambda xi: tree_ensemble.evaluate(xi[:], m+1), axis=1)
        y_test_ensemble_pred = test_data.apply(lambda xi: tree_ensemble.evaluate(xi[:], m+1), axis=1)

        train_mean_loss = np.mean((y_train - y_train_ensemble_pred) ** 2)
        test_mean_loss = np.mean((y_test - y_test_ensemble_pred) ** 2)

        print('Add tree number {}'.format(m+1))
        print('Train mean loss is: {}'.format(train_mean_loss))
        print('Test mean loss is: {}'.format(test_mean_loss))

    train_data[label_name] = y_train

    return tree_ensemble




