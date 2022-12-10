from data_utils import parse_data
from gbrt_algorithm import gbrt
from feature_selection import ensemble_feature_importance


class hparams(object):
    def __init__(self, num_trees, max_depth, min_node_size, weight_decay, sub_samp):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_node_size = min_node_size
        self.weight_decay = weight_decay
        self.sub_samp = sub_samp


train_dataset, test_dataset = parse_data('data/train.csv')
params = hparams(num_trees=10, max_depth=3, min_node_size=0, weight_decay=0.9, sub_samp=1)
model = gbrt(train_data=train_dataset.data, test_data=test_dataset.data, label_name=train_dataset.label_name, params=params)

features_dict = ensemble_feature_importance(test_dataset, train_dataset.label_name, model)
