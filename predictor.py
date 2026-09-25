import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# splitting the data
def split_data(data: pd.DataFrame):
    """Splits the dataset into the training and testing sets for the initial football dataset that spans 2005-2025.
    The original training set will contain the data from 2005-2023 and the testing set will test over the 2024-2025 seasons.
    """
    cutoff = (data["Season"].values == "2023-2024").argmax()
    data_dropped = data.drop("Season", axis=1)

    training_data = data_dropped.iloc[:cutoff, :]
    testing_data = data_dropped.iloc[cutoff:, :]

    # target_cols = ["target_A", "target_D", "target_H"]

    train_data_features = training_data.drop("target", axis=1)
    train_data_target = training_data["target"].to_numpy()

    testing_data_features = testing_data.drop("target", axis=1)
    testing_data_target = testing_data["target"].to_numpy()

    return train_data_features, train_data_target, testing_data_features, testing_data_target

def preproc(df: pd.DataFrame):
    # drop unnecessary string columns
    drop_cols = ["Date", "HomeTeam", "AwayTeam", "HTR"]
    df = df.drop(columns=drop_cols)

    # # one-hot encode the target column
    # dummies = pd.get_dummies(data=df["target"], prefix="target", dtype=float)
    # df = pd.concat([df, dummies], axis=1)
    # df = df.drop("target", axis=1)

    return df

def feature_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def main():
    # checking the data and value counts for the target variable
    initial_fd = pd.read_csv("./data/fd_features_2005_2025.csv")
    print(f"D: {len(initial_fd[initial_fd["target"] == "D"])}")
    print(f"A: {len(initial_fd[initial_fd["target"] == "A"])}")
    print(f"H: {len(initial_fd[initial_fd["target"] == "H"])}")

    # preprocess data
    pp_df = preproc(initial_fd)

    # split the data into training and testing sets and scale for the algorithm
    X_train, y_train, X_test, y_test = split_data(pp_df)
    X_train_scaled, X_test_scaled = feature_scale(X_train, X_test)

    # train some classifiers
    # random forest
    rfc = RandomForestClassifier()
    rfc.fit(X_train_scaled, y_train)

    # evaluation
    accuracy = rfc.score(X_test_scaled, y_test)
    print(f"Random Forest Accuracy: {accuracy}")


if __name__ == "__main__":
    main()