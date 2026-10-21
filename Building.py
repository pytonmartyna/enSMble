import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, f1_score, recall_score, accuracy_score,
                             precision_score, mean_squared_error)


def crossvalidate_and_tune(model_type: str, df_train_selected: pd.DataFrame,
                           y_train: pd.Series, return_best_score: bool=False, return_cv_results : bool=False):
    """
        This function performs cross-validation combined with hyperparameters tuning using RandomizedSearchCV class.

        Args:
            model_type (str): Type of machine learning model to be validated and tuned.
            df_train_selected (pd.DataFrame): Post feature selection training dataset.
            y_train (pd.Series): Vector of class labels for training dataset.
            return_best_score (bool): An argument determining whether the best result achieved by the model
            should appear in the output dictionary.
            return_cv_results (bool): An argument determining whether detailed validation results
            should appear in the output dictionary.

        Returns:
            dict: Dictionary containing the best set of hyperparameters and optionally other results.

    """
    available_models = {"randomforest": RandomForestClassifier,
                         "svm": SVC}
    hyperparam_grids = {"randomforest": {"n_estimators": np.arange(100, 300, 25),
                                         "max_depth":np.arange(5, 20, 3),
                                         "min_samples_split":np.arange(5, 20, 3),
                                         "min_samples_leaf":np.arange(1, 10, 2),
                                         "max_features": ['sqrt', 'log2', 0.01, 0.03, 0.05, 0.1]},
                        "svm":{"C":[0.01, 0.1, 1.0, 10.0],
                               "kernel":['linear', 'rbf'],
                               "gamma":['scale', 'auto', 0.001, 0.01, 0.1, 1.0]}}
    model_type_normalized = model_type.lower().replace(" ", "").replace("_", "").replace("-", "")
    if len(df_train_selected) == len(y_train):
        if model_type_normalized in list(available_models.keys()):
            rscv = RandomizedSearchCV(estimator = available_models[model_type_normalized](class_weight = "balanced",
                                      random_state = 50),
                                      param_distributions = hyperparam_grids[model_type_normalized],
                                      n_iter = 10, n_jobs = -1, cv = StratifiedKFold(n_splits = 10),
                                      scoring = "roc_auc", return_train_score = True, random_state = 50)
            rscv.fit(df_train_selected, y_train)
            return {"best_params": rscv.best_params_,
                    "best_score": rscv.best_score_ if return_best_score == True else None,
                    "cv_results": rscv.cv_results_ if return_cv_results == True else None}
        else:
            raise ValueError("Unsupported model type named " + model_type + " was entered. Try again.")
    else:
        raise ValueError("Given dataset and vector of class labels are not of the same length. Try again.")



def build_and_train(model_type:str, df_train: pd.DataFrame, y_train: pd.Series, **kwargs):
    """
        This function builds and performs a training of target machine learning model.

        Args:
            model_type (str): Type of machine learning model to build and train.
            df_train (pd.DataFrame): Post selection training dataset.
            y_train (pd.Series): Vector of class labels for training dataset.
            **kwargs (dict): Dictionary of input arguments passed to build the model, i.e. its corresponding hyperparameters.

        Returns:
            model (BaseEstimator): Result machine learning model of type specified in the input arguments.
    """
    available_models = {"randomforest": lambda **kwargs : RandomForestClassifier(
                                                          class_weight = "balanced", random_state = 50, **kwargs
                                                          ),
                         "svm": lambda **kwargs: SVC(class_weight = "balanced", probability = True,
                                                      random_state = 50, **kwargs)}
    model_type_normalized = model_type.lower().replace(" ", "").replace("_", "").replace("-", "")
    if model_type_normalized in list(available_models.keys()):
        try:
            model = available_models[model_type_normalized](**kwargs)
        except (TypeError, KeyError) as e:
            raise ValueError("Invalid dictionary content: " + str(e))
    else:
        raise ValueError("Unsupported model type: " + model_type + " was entered. Try again.")
    if len(df_train) == len(y_train):
        model.fit(df_train, y_train)
        return model
    else:
        raise ValueError("Given dataset and vector of class labels are not of the same length. Try again.")


def test_and_evaluate(metrics: list, model: BaseEstimator, df_test: pd.DataFrame,
                      y_test: pd.Series, return_y_pred: bool = False):
    """
        This function performs model testing and evaluation using the specified input metrics.

        Args:
             metrics (list): List of metrics used for model evaluation.
             model (BaseEstimator): Pre-built machine learning model to test and evaluate.
             df_test (pd.DataFrame): Test dataset.
             y_test (pd.Series): Vector of class labels for test dataset to be predicted by the input model.
             return_y_pred (bool): An argument determining whether predicted vector of class labels
             should appear in the output dictionary.

        Returns:
            dict: Dictionary containing list of scores received by a model via evaluation
            and optionally a predicted vector of class labels.
    """
    if len(df_test) == len(y_test):
        y_pred = model.predict(df_test)
    else:
        raise ValueError("Given dataset and vector of class labels are not of the same length. Try again.")
    scoring = {}
    available_metrics = {"accuracy" : accuracy_score,
                         "f1" : f1_score,
                         "rocauc" : roc_auc_score,
                         "mcc" : matthews_corrcoef,
                         "precision": precision_score,
                         "recall": recall_score,
                         "mse": mean_squared_error}
    for metric in metrics:
        metric_normalized = metric.lower().replace(" ", "").replace("_", "").replace("-", "").strip()
        if metric_normalized in list(available_metrics.keys()) and metric_normalized != "rocauc":
            scoring[metric_normalized] = available_metrics[metric_normalized](y_test, y_pred)
        elif metric_normalized in list(available_metrics.keys()) and metric_normalized == "rocauc":
            y_pred_roc = model.predict_proba(df_test)[:, 1]
            scoring[metric_normalized] = available_metrics[metric_normalized](y_test, y_pred_roc)
        else:
            raise ValueError("Unsupported metric named " + metric_normalized + " appeared on the list. Try again.")
    return {"scoring": scoring,
            "y_pred": y_pred if return_y_pred == True else None}

















