import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

    return df

def feature_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler()
    other_scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def fit_model_search_cv(base_model, param_grid, X_train_scaled, y_train):
    model_search_cv = RandomizedSearchCV(
        estimator=base_model, 
        param_distributions=param_grid,
        n_iter=10,
        n_jobs=-1,
        cv=5,
        verbose=1,
        random_state=4,
        return_train_score=True
    )

    return model_search_cv.fit(X_train_scaled, y_train)

def evaluate_model(trained_model, X_test_scaled, y_test):
    best_model = trained_model.best_estimator_
    model_name = best_model.__class__.__name__
    print("-"*65)
    print(f"EVALUATING {model_name}")
    print("-"*65)
    best_params = trained_model.best_params_
    results = trained_model.cv_results_
    results_df = pd.DataFrame(results)
    best_index = trained_model.best_index_
    best_score = trained_model.best_score_
    avg_training_score = sum(results_df["mean_train_score"])/len(results_df)
    avg_testing_score = sum(results_df["mean_test_score"])/len(results_df)
    print(f"The best parameters of this {model_name} model are: {best_params}\n")
    print(f"The search's results are printed here:\n{results_df}\n")
    print(f"The table's data for the best parameters of the {model_name} model are:\n{results_df.iloc[best_index, :]}\n")
    print(f"The average training score for this {model_name} model is: {avg_training_score}\n")
    print(f"The average testing score for this {model_name} model is: {avg_testing_score}\n")
    print(f"The best score of the {model_name} model during training was: {best_score}\n")

    val_accuracy_score = best_model.score(X_test_scaled, y_test)
    print(f"The {model_name} best model's accuracy on unseen data is: {val_accuracy_score}\n")
    print(f"The {model_name} model selected from {len(set(y_test))} classes that included: {set(y_test)}\n")
    preds = best_model.predict(X_test_scaled)
    conf_mat = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["A", "D", "H"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show(block=False)

    return best_model, best_params

def train_diff_classifiers(estimator, b_model_params: dict, X_train_scaled, y_train):
    # one v one classifier
    # ** will unpack the dictionary of parameters to be set within the base estimator
    spec_model = estimator.set_params(**b_model_params)
    spec_model_seeded = spec_model.set_params(random_state=4)
    ovo_clf = OneVsOneClassifier(spec_model_seeded).fit(X_train_scaled, y_train)

    # one v rest classifier
    ovr_clf = OneVsRestClassifier(spec_model_seeded).fit(X_train_scaled, y_train)

    # output code classifier
    output_code_clf = OutputCodeClassifier(spec_model_seeded).fit(X_train_scaled, y_train)

    return ovo_clf, ovr_clf, output_code_clf

def evaluate_multiclass_model(mult_model, X_test_scaled, y_test):
    mult_model_name = mult_model.__class__.__name__
    print("-"*65)
    print(f"EVALUATING {mult_model_name}")
    print("-"*65)
    val_accuracy_score = mult_model.score(X_test_scaled, y_test)
    print(f"The {mult_model_name} best model's accuracy on unseen data is: {val_accuracy_score}\n")

    print(f"The {mult_model_name} model selected from {len(set(y_test))} classes that included: {set(y_test)}\n")
    preds = mult_model.predict(X_test_scaled)
    conf_mat = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["A", "D", "H"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {mult_model_name}")
    plt.show(block=False)

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

    print(f"X_train_len: {len(X_train_scaled)}")
    print(f"X_test_len: {len(X_test_scaled)}")
    print(f"y_train_len: {len(y_train)}")
    print(f"y_test_len: {len(y_test)}")

    # train some classifiers
    # rand forest
    rfc = RandomForestClassifier(random_state=4)
    rf_param_grid = {
        "n_estimators": [500, 1000, 2000],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 10, 20],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", None]
    }
    # model_search_cv = fit_model_search_cv(rfc, rf_param_grid, X_train_scaled, y_train)
    # best_rfc, rfc_b_params = evaluate_model(model_search_cv, X_test_scaled, y_test)

    # linear support vector machine
    svm = LinearSVC(random_state=4)
    svm_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100, 1000],
        "tol": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "multi_class": ["ovr", "crammer_singer"]
    }
    model_search_cv = fit_model_search_cv(svm, svm_param_grid, X_train_scaled, y_train)
    best_linear_svm, l_svm_b_params = evaluate_model(model_search_cv, X_test_scaled, y_test)

    # support vector machine
    svc = SVC(random_state=4)
    svc_param_grid = {
        "C": [0.01, 0.1, 1, 10, 100, 1000],
        "kernel": ["poly", "rbf", "sigmoid"],
        "degree": [2, 3],
        "gamma": ["scale", "auto"],
        "tol": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
        "class_weight": ["balanced", None]
    }
    # model_search_cv = fit_model_search_cv(svc, svc_param_grid, X_train_scaled, y_train)
    # best_svc_model, svm_b_params = evaluate_model(model_search_cv, X_test_scaled, y_test)

    # # naive bayes
    # mnb = MultinomialNB()
    # mnb.fit(X_train_scaled, y_train)
    # print(f"The accuracy of the multinomial naive bayes model is: {mnb.score(X_test_scaled, y_test)}")

    # K-Neighbors Classifier
    knn = KNeighborsClassifier()
    knn_param_grid = {
        "n_neighbors": [3, 5, 7, 10, 20],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "brute"],
        "p": [1, 2]
    }

    # model_search_cv = fit_model_search_cv(knn, knn_param_grid, X_train_scaled, y_train)
    # best_knn_model, knn_b_params = evaluate_model(model_search_cv, X_test_scaled, y_test)

    # train logistic regression or hist gradient boosting models
    

    # look at changing the strategy of classification between one v one, one v rest, and an output code classifier
    ovo_clf, ovr_clf, oc_clf = train_diff_classifiers(best_linear_svm, l_svm_b_params, X_train_scaled, y_train)
    evaluate_multiclass_model(ovo_clf, X_test_scaled, y_test)
    evaluate_multiclass_model(ovr_clf, X_test_scaled, y_test)
    evaluate_multiclass_model(oc_clf, X_test_scaled, y_test)

    

    # keeping all plots open at the end of the script
    plt.show()


if __name__ == "__main__":
    main()