import pandas as pd
import numpy as np


def split_data(data, train_percent=0.8):
    train_size = int(len(data) * train_percent)
    train_indices = np.random.choice(data.index, train_size, replace=False)

    train = data.loc[data.index.isin(train_indices)]
    test = data.loc[~data.index.isin(train_indices)]

    return train, test


def parse_data(path):
    df = pd.read_csv(path, index_col=0)
    df = df[~df.SalePrice.isnull()]
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df.MSSubClass.astype(object)  # categorical feature with numerical values

    df_train, df_test = split_data(df)

    train_dataset = TrainDataSet(df_train)
    test_dataset = TestDataSet(df_test, train_dataset.categorical_columns_coding_map,
                               train_dataset.numerical_columns_means)

    return train_dataset, test_dataset


def encode_categorical_columns(data, encoding_map):
    categorical_columns = data.loc[:, data.dtypes == object].columns

    for col_name in categorical_columns:
        col_encoding_map = encoding_map[col_name]
        try:
            col_null_encoding = encoding_map[col_name].loc['null_value']
        except KeyError:
            col_null_encoding = col_encoding_map.max()

        encoded_col = data[col_name].map(col_encoding_map)
        encoded_col.fillna(col_null_encoding, inplace=True)

        data[col_name] = encoded_col


def fillna_numerical_columns(data, imputation_values):
    data.fillna(imputation_values, inplace=True)


class TrainDataSet:
    def __init__(self, data, label_name='SalePrice'):
        self.data = data
        self.label_name = label_name

        self.categorical_columns_coding_map = self.get_categorical_columns_coding_map()
        self.numerical_columns_means = self.get_numerical_columns_means()

        encode_categorical_columns(self.data, self.categorical_columns_coding_map)
        fillna_numerical_columns(self.data, self.numerical_columns_means)

    def get_categorical_columns_coding_map(self):
        categorical_columns = self.data.loc[:, self.data.dtypes == object].columns

        categorical_code_map = {}
        for column in categorical_columns:
            column_code = self.data[[column, self.label_name]].fillna('null_value').groupby(column).mean().rank()
            categorical_code_map[column] = column_code[self.label_name]

        return categorical_code_map

    def get_numerical_columns_means(self):
        return self.data._get_numeric_data().mean()


class TestDataSet:
    def __init__(self, data, categorical_columns_coding_map, numerical_columns_means):
        self.data = data

        encode_categorical_columns(self.data, categorical_columns_coding_map)
        fillna_numerical_columns(self.data, numerical_columns_means)

