import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import typing

def preprocess_features(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_labels = train_dataframe.xs('class', axis='columns') == 0
    remove_class_columns(train_dataframe)
    test_labels = test_dataframe.xs('class', axis='columns') == 0
    remove_class_columns(test_dataframe)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_dataframe), columns = train_dataframe.columns)
    X_test = pd.DataFrame(scaler.transform(test_dataframe), columns = test_dataframe.columns)

    return X_train, X_test, train_labels, test_labels

def minmax_preprocess_features(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_labels = train_dataframe.xs('class', axis='columns') == 0
    remove_class_columns(train_dataframe)
    test_labels = test_dataframe.xs('class', axis='columns') == 0
    remove_class_columns(test_dataframe)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_dataframe), columns = train_dataframe.columns)
    X_test = pd.DataFrame(scaler.transform(test_dataframe), columns = test_dataframe.columns)

    return X_train, X_test, train_labels, test_labels

def minmax_preprocess_denoised_features(denoised_train_dataframe: pd.DataFrame) -> (pd.DataFrame):
    scaler = MinMaxScaler()
    return scaler.fit_transform(denoised_train_dataframe)

def minmax_preprocess_z_features(z_feature_cols: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit_predict(z_feature_cols)

def remove_class_columns(dataframe: pd.DataFrame) -> None:
    for column in dataframe.columns:
        if (column[0:5] == 'class'):
            dataframe.drop(column, axis='columns', inplace=True)