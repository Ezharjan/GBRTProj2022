import functools
import operator


def tree_feature_importance(dataset, label_name, tree):
    features_dict = {}
    root = tree.get_root()
    this_level = [(root, dataset)]
    while this_level:
        next_level = list()
        for node, data in this_level:
            if node.is_terminal():
                continue
            # Save for the next level
            RL_data = data[data.iloc[:, node.j] <= node.s]
            RR_data = data[data.iloc[:, node.j] > node.s]
            next_level.append((node.left_descendant, RL_data))
            next_level.append((node.right_descendant, RR_data))
            # calculate needed constatns
            c_bef = node[label_name].mean()
            c_L = RL_data[label_name].mean()
            c_R = RR_data[label_name].mean()
            if node.j not in features_dict.keys():
                features_dict[node.j] = 0

            features_dict[node.j] += \
                functools.reduce(operator.add, [(row[label_name] - c_bef) ** 2 for _, row in data.iterrows()]) + \
                functools.reduce(operator.add, [(row[label_name] - c_L) ** 2 for _, row in RL_data.iterrows()]) + \
                functools.reduce(operator.add, [(row[label_name] - c_R) ** 2 for _, row in RR_data.iterrows()])

            this_level = next_level
    return features_dict


def ensemble_feature_importance(dataset, label_name, tree_ensemble):
    ensemble_features_dict = {}
    for tree in tree_ensemble.trees:
        tree_features_dict = tree_feature_importance(dataset, label_name, tree)
        for feature_id, score in tree_features_dict.items():
            if feature_id not in ensemble_features_dict.keys():
                ensemble_features_dict[feature_id] = 0
            ensemble_features_dict[feature_id] += score
    ensemble_features_dict = sorted(ensemble_features_dict.items(), key=operator.itemgetter(1), reverse=True)
    most_imp_val = ensemble_features_dict.values()[0]
    ensemble_features_dict = {k: float(v / most_imp_val) for k, v in ensemble_features_dict.items()}
    return ensemble_features_dict