import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from numbers import Number
import os
from upsetplot import from_contents, UpSet

def feature_importances_plot(feature_importances: np.ndarray, features: list, path_to_save: str):
    """
        This function generates feature importances plot applicable for cases when machine learning model supports
        feature_importances_ attribute.

        Args:
            feature_importances (np.ndarray): An array of feature importances.
            features (list): List of features in the dataset.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    fimportances_matched = pd.DataFrame({"feature": features})
    fimportances_matched.set_index("feature", inplace=True)
    fimportances_matched["importance"] = feature_importances
    fimportances_matched.sort_values(by="importance", ascending=False, inplace=True)
    top_fimportances = fimportances_matched.iloc[0:21]
    sns.set_style("whitegrid")
    fi_plot = sns.barplot(data=top_fimportances, x="feature", y="importance", legend="auto")
    fi_plot.set_title("Feature Importances Plot")
    plt.xticks(rotation=45)
    filetitle = f"{fi_plot.get_title().replace(" ", "_").lower()}.png"
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()

def feature_corr_heatmap(df_train_selected: pd.DataFrame, path_to_save: str):
    """
        This function generates features correlation heatmap.

        Args:
            df_train_selected (pd.DataFrame): Post-selection training dataset.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    fcorr_matrix = df_train_selected.corr()
    width = max(14, len(fcorr_matrix) * 0.5)
    height = max(12, len(fcorr_matrix) * 0.5)
    plt.figure(figsize = (width, height))
    sns.set(font_scale = min(1.0, 200 / len(fcorr_matrix)))
    fcorr_heatmap = sns.heatmap(data = fcorr_matrix, cmap = "coolwarm", annot = True, fmt = ".2f", cbar = True)
    fcorr_heatmap.set_title("Selected Features Correlation Heatmap")
    filetitle = f"{fcorr_heatmap.get_title().replace(" ", "_").lower()}.png"
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()

def hyperparams_validation_curve(cv_results: dict, hyperparams: list, path_to_save: str):
    """
        This function generates validation curve plot applicable for cases when
        the cv_results parameter follows the convention of Scikit-learn class objects such as RandomizedSearchCV.

        Args:
            cv_results (dict): Dictionary of detailed tuning results,
            the keys of which will be used to construct graphs illustrating the impact of hyperparameter values
            on model performance.
            hyperparams (list): List of hyperparameters for which validation charts are to be created.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    try:
        hparams_to_plot = {}
        for param in hyperparams:
            param_normalized = "param_" + param.strip()
            if param_normalized in list(cv_results.keys()):
                hparams_to_plot[param_normalized] = cv_results[param_normalized]
            else:
                print(f"Invalid hyperparameter {param_normalized}! Try again.")
        mean_train_scores = cv_results["mean_train_score"]
        mean_test_scores = cv_results["mean_test_score"]
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(5*len(list(hparams_to_plot.keys())), 5))
        for i, key in enumerate(list(hparams_to_plot.keys())):
            if all(isinstance(value, Number) for value in hparams_to_plot[key]):
                uniques = sorted(set(hparams_to_plot[key]))
            else:
                uniques = list(set(hparams_to_plot[key]))
            mean_train_per_param = []
            mean_test_per_param = []
            for item in uniques:
                indices = [idx for idx, value in enumerate(hparams_to_plot[key]) if str(value) == str(item)]
                mean_train_per_param.append(np.mean([mean_train_scores[idx] for idx in indices]))
                mean_test_per_param.append(np.mean([mean_test_scores[idx] for idx in indices]))
            axes = plt.subplot(1, len(list(hparams_to_plot.keys())), i + 1)
            plt.plot(uniques, mean_train_per_param, label = "Training Curve", marker = 'o')
            plt.plot(uniques, mean_test_per_param, label = "Validation Curve", marker = 'o')
            plt.grid(True)
            param_title = key.replace("param_", "").replace("_", " ").title()
            plt.title(f"{param_title} Validation Curve")
            plt.xlabel(f"{param_title} values")
            plt.ylabel("Mean CV Scores")
            plt.legend()
            plt.tight_layout()
        filetitle = "hyperparams_validation_curve.png"
        try:
            plt.savefig(os.path.join(path_to_save, filetitle))
        except (OSError, FileNotFoundError) as e:
            print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
        plt.show()
    except (KeyError, TypeError) as e:
        raise ValueError("Invalid arguments. Check dictionary and list content. " + str(e))


def class_prediction_error_plot(y_test: pd.Series, y_pred: pd.Series, path_to_save: str):
    """
        This function generates a Class Prediction Error Plot.

        Args:
            y_test (pd.Series): Actual vector of class labels for testing dataset.
            y_pred (pd.Series): Predicted vector of class labels for testing dataset.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    if len(y_test) != len(y_pred):
        raise ValueError("Vectors of actual and predicted class labels are not of the same length. Try again!")
    conf_matrix = confusion_matrix(y_test, y_pred)
    confusion_matrix_percentage = conf_matrix/conf_matrix.sum(axis = 1, keepdims = True)
    c_p_e_plot = sns.heatmap(data = confusion_matrix_percentage, cmap = "coolwarm", cbar = True, annot = True, fmt = ".2%")
    c_p_e_plot.set_title("Class Prediction Error Plot")
    filetitle = f"{c_p_e_plot.get_title().replace(" ", "_").lower()}.png"
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()

def class_imbalance_barplot(y: pd.Series, path_to_save: str):
    """
        This function generates Class Imbalance Bar Plot.

        Args:
            y (pd.Series): Vector of class labels for the complete dataset.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    class_names = ["class " + str(idx) for idx in y.value_counts().index]
    class_counts = y.value_counts().values
    plt.style.use("fivethirtyeight")
    figure, axes = plt.subplots(figsize = (8,8))
    axes.bar(class_names, class_counts)
    axes.grid(True)
    axes.set_title("Class Imbalance Bar Plot")
    axes.set_xlabel("Class Names")
    axes.set_ylabel("Number of Samples")
    filetitle = f"{axes.get_title().replace(" ", "_").lower()}.png"
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()

def class_imbalance_scatterplot(df_train_selected: pd.DataFrame, y_train: pd.Series,
                                n_components: int, path_to_save: str):
    """
        This function generates Class Imbalance Scattered Plot using PCA in 2D or 3D space.

        Args:
            df_train_selected (pd.DataFrame): Post-selection dataset.
            y_train (pd.Series): Vector of class labels for the input dataset.
            n_components (int): Number of components for PCA to extract from the set of features
            in order to project data onto a space with a number of dimensions equal to n_components parameter.
            This dimensionality reduction combined with preserving as much variance as possible
            can be done thanks to the selected n_components.
            path_to_save (str): Path where the generated graph should be saved.

        Returns:
            None.
    """
    if len(df_train_selected) != len(y_train):
        raise ValueError("Given dataset and vector of class labels are not of the same length. Try again.")
    pca = PCA(n_components = n_components)
    df_pca = pca.fit_transform(df_train_selected)
    if n_components == 2:
        figure, axes = plt.subplots(figsize = (8,8))
        axes.scatter(df_pca[:, 0], df_pca[:, 1], c = y_train, cmap = "coolwarm", alpha = 0.7)
        axes.set_xlabel("1st Component")
        axes.set_ylabel("2nd Component")
        axes.set_title("Class Imbalance Scatter Plot in PCA 2D Space")
        filetitle = f"{axes.get_title().replace(" ", "_"). lower()}.png"
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        plot_3d = plt.figure().add_subplot(111, projection="3d")
        plot_3d.scatter(df_pca[:, 0], df_pca[:, 1], df_pca[:, 2], c = y_train, cmap = "coolwarm", alpha = 0.7)
        plot_3d.set_xlabel("1st Component")
        plot_3d.set_ylabel("2nd Component")
        plot_3d.set_zlabel("3rd Component")
        plot_3d.set_title("Class Imbalance Scatter Plot in PCA 3D Space")
        filetitle = f"{plot_3d.get_title().replace(" ", "_"). lower()}.png"
    else:
        raise ValueError("Wrong number of dimensions, only 2D and 3D supported, try again!")
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()

def upset_chart(base_methods_dict: dict, path_to_save: str):
    """
        This function generates UpSet chart including intersections between different combinations of
        base methods used in ensemble feature selection.

         Args:
             base_methods_dict (dict): Dictionary containing pairs, where the keys are the names
             of the selection methods used in the ensemble and the values are the lists of features selected by them.
             path_to_save (str):  Path where the generated graph should be saved.

        Returns:
            None.
    """
    data = from_contents(base_methods_dict)
    plt.figure(figsize=(12, 6))
    UpSet(data, show_counts=True, sort_by="degree").plot()
    plt.title("UpSet chart - Ensemble Feature Selection Using 6 Base Methods")
    filetitle = "upset_chart.png"
    try:
        plt.savefig(os.path.join(path_to_save, filetitle))
    except (OSError, FileNotFoundError) as e:
        print("Incorrect Path: " + path_to_save + ". Make sure given path exists!" + str(e))
    plt.show()



