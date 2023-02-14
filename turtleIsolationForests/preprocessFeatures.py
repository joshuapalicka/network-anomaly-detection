import pandas as pd
from sklearn.preprocessing import StandardScaler
import typing

def preprocess_features(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    train_labels = train_dataframe.xs('class', axis='columns')
    remove_class_columns(train_dataframe)
    test_labels = test_dataframe.xs('class', axis='columns')
    remove_class_columns(test_dataframe)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_dataframe), columns = train_dataframe.columns)
    X_test = pd.DataFrame(scaler.transform(test_dataframe), columns = test_dataframe.columns)

    return X_train, X_test, train_labels, test_labels
    

def remove_class_columns(dataframe: pd.DataFrame) -> None:
    for column in dataframe.columns:
        if (column[0:5] == 'class'):
            dataframe.drop(column, axis='columns', inplace=True)