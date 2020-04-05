"""
This program will load the train and test data provided by Kaggle.
It will also plot a correlation matrix of the training features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def feature_correlation() -> None:
    """
    Plots the correlation matrix of the training features
    """
    le = LabelEncoder() # used to convert string values to numerical values for training the model
    train = pd.read_csv("../titanic/train.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(train.columns.values):
        if col_name == "Age" and "Pclass" in list(train.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            train[col_name] = train.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            train[col_name].astype('float64')
            continue
        train[col_name] = le.fit_transform(train[col_name].astype("str"))

    matrix = train.corr()
    plt.figure(figsize=(20, 14))
    sns.set(font_scale=2.0)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 20}, cmap="YlGnBu",)
    plt.show()

def train_data(drop_columns = []) -> dict:
    """
    This separates the features and labels of the training data
    Note that we are using 'Survived' as the label

    Parameters
    -------------
    drop_columns: list-like, optional
        list-like object of all the columns to drop

    Returns
    -------------
    dict
        dictionary of feature and label arrays
    """
    le = LabelEncoder() # used to convert string values to numerical values for training the model
    # The .drop() portion is to remove features that will not help in predicting Survival
    train = pd.read_csv("../titanic/train.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(train.columns.values):
        if col_name == "Age" and "Pclass" in list(train.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            train[col_name] = train.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            train[col_name].astype('float64')
            continue
        train[col_name] = le.fit_transform(train[col_name].astype("str"))
    train.to_csv("../titanic/toya_train.csv", index = False)
    train.drop(columns = drop_columns, inplace = True)
    print("\nTraining Set Info:")
    print(train.info())
    train_labels = train["Survived"]
    train_features = train.drop(columns=["Survived"])
    return {"features": train_features.values, "labels": train_labels.values}


def test_data(drop_columns = []) -> np.ndarray:
    """
    imports a set of test features for testing performance
    of your ML model. Note that this should be strictly used
    for testing and NOT validating your model

    Parameters
    -------------
    drop_columns: list-like, optional
        list-like object of all the columns to drop

    Returns
    -------------
    np.ndarray
        array of features
    """

    le = LabelEncoder() # used to convert string values to numerical values for training the model
    # The .drop() portion is to remove features that will not help in predicting Survival
    test = pd.read_csv("../titanic/test.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(test.columns.values):
        if col_name == "Age" and "PClass" in list(test.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            test[col_name] = test.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            test[col_name].fillna(8, inplace = True)
            test[col_name].astype('float64')
            continue
        test[col_name] = le.fit_transform(test[col_name].astype("str"))
    test.to_csv("../titanic/toya_test.csv", index = False)
    test.drop(columns = drop_columns, inplace = True)
    print("\nTesting Set Info:")
    print(test.info())
    return test.values

def new_feature_correlation() -> None:
    """
    Plots the correlation matrix of the training features
    Cabin, SibSp, and Parch have been transformed into true / false
    values (1 for true, 0 for false)
    """
    le = LabelEncoder() # used to convert string values to numerical values for training the model
    train = pd.read_csv("../titanic/train.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(train.columns.values):
        if col_name == "Age" and "Pclass" in list(train.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            train[col_name] = train.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            train[col_name].astype('float64')
            continue
        if col_name in ["SibSp", "Parch"]:
            train[col_name] = train.apply(
                lambda row: 1 if row[col_name] > 0 else 0, axis = 1
            )
        if col_name == "Cabin":
            train[col_name] = train.apply(
                lambda row: 0 if pd.isna(row[col_name]) else 1, axis = 1
            )
        train[col_name] = le.fit_transform(train[col_name].astype("str"))

    matrix = train.corr()
    plt.figure(figsize=(20, 14))
    sns.set(font_scale=2.0)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 20}, cmap="YlGnBu",)
    plt.show()

def new_train_data(drop_columns = []) -> dict:
    """
    This separates the features and labels of the training data
    Note that we are using 'Survived' as the label
    Cabin, SibSp, and Parch have been transformed into true / false
    values (1 for true, 0 for false)

    Parameters
    -------------
    drop_columns: list-like, optional
        list-like object of all the columns to drop

    Returns
    -------------
    dict
        dictionary of feature and label arrays
    """
    le = LabelEncoder() # used to convert string values to numerical values for training the model
    # The .drop() portion is to remove features that will not help in predicting Survival
    train = pd.read_csv("../titanic/train.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(train.columns.values):
        if col_name == "Age" and "Pclass" in list(train.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            train[col_name] = train.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            train[col_name].astype('float64')
            continue
        if col_name in ["SibSp", "Parch"]:
            train[col_name] = train.apply(
                lambda row: 1 if row[col_name] > 0 else 0, axis = 1
            )
        if col_name == "Cabin":
            train[col_name] = train.apply(
                lambda row: 0 if pd.isna(row[col_name]) else 1, axis = 1
            )
        train[col_name] = le.fit_transform(train[col_name].astype("str"))
    train.to_csv("../titanic/toya_new_train.csv", index = False)
    train.drop(columns = drop_columns, inplace = True)
    print("\nNew Training Set Info:")
    print(train.info())
    train_labels = train["Survived"]
    train_features = train.drop(columns=["Survived"])
    return {"features": train_features.values, "labels": train_labels.values}


def new_test_data(drop_columns = []) -> np.ndarray:
    """
    imports a set of test features for testing performance
    of your ML model. Note that this should be strictly used
    for testing and NOT validating your model
    Cabin, SibSp, and Parch have been transformed into true / false
    values (1 for true, 0 for false)

    Parameters
    -------------
    drop_columns: list-like, optional
        list-like object of all the columns to drop

    Returns
    -------------
    np.ndarray
        array of features
    """

    le = LabelEncoder() # used to convert string values to numerical values for training the model
    # The .drop() portion is to remove features that will not help in predicting Survival
    test = pd.read_csv("../titanic/test.csv", encoding = "UTF-8", delimiter = ",")
    for col_name in list(test.columns.values):
        if col_name == "Age" and "PClass" in list(test.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            test[col_name] = test.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            test[col_name].fillna(8, inplace = True)
            test[col_name].astype('float64')
            continue
        if col_name in ["SibSp", "Parch"]:
            test[col_name] = test.apply(
                lambda row: 1 if row[col_name] > 0 else 0, axis = 1
            )
        if col_name == "Cabin":
            test[col_name] = test.apply(
                lambda row: 0 if pd.isna(row[col_name]) else 1, axis = 1
            )
        test[col_name] = le.fit_transform(test[col_name].astype("str"))
    test.to_csv("../titanic/toya_new_test.csv", index = False)
    test.drop(columns = drop_columns, inplace = True)
    print("\nNew Testing Set Info:")
    print(test.info())
    return test.values

def balanced_train_data(drop_columns = []) -> dict:
    """
    This separates the features and labels of the training data
    Note that we are using 'Survived' as the label.
    Balances the training data by duplicating the survived passengers

    Parameters
    -------------
    drop_columns: list-like, optional
        list-like object of all the columns to drop

    Returns
    -------------
    dict
        dictionary of feature and label arrays
    """
    le = LabelEncoder() # used to convert string values to numerical values for training the model
    # The .drop() portion is to remove features that will not help in predicting Survival
    train = pd.read_csv("../titanic/train.csv", encoding = "UTF-8", delimiter = ",")
    survived = train[train["Survived"] == 1]
    train = train.append(survived)
    for col_name in list(train.columns.values):
        if col_name == "Age" and "Pclass" in list(train.columns.values):
            # we will fill all the NaN ages with an age based on passenger class
            train[col_name] = train.apply(
                lambda row: 37 if np.isnan(row[col_name]) and row["Pclass"] == 1 else (
                    29 if np.isnan(row[col_name]) and row["Pclass"] == 2 else (
                        24 if np.isnan(row[col_name]) and row["Pclass"] == 3 else(
                            row[col_name]
                        )
                    )
                ),
                axis = 1
            )
        if col_name == "Fare":
            train[col_name].astype('float64')
            continue
        train[col_name] = le.fit_transform(train[col_name].astype("str"))
    train.to_csv("../titanic/toya_balanced_train.csv", index = False)
    train.drop(columns = drop_columns, inplace = True)
    print("\nTraining Set Info:")
    print(train.info())
    train_labels = train["Survived"]
    train_features = train.drop(columns=["Survived"])
    return {"features": train_features.values, "labels": train_labels.values}
