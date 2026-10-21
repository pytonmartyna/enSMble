import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from skfeature.function.similarity_based.fisher_score import fisher_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from skrebate import ReliefF
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def u_test(data: pd.DataFrame, target_col: pd.Series):
    """
        This method performs Mann-Whitney U-Test on dataset with two distinctive classes.

        Args:
             data (pd.DataFrame): Input training dataset.
             target_col (pd.Series): Vector of class labels for training dataset.

        Returns:
            results_ndarray (numpy.ndarray): Array of p value calculated for each feature in the input dataset.
            This object is needed to perform U-Test based feature selection.
    """
    results = []
    group1_inds = target_col[target_col == 0].index
    group2_inds = target_col[target_col == 1].index
    for feature in data.columns:
        group1 = data.loc[group1_inds, feature]
        group2 = data.loc[group2_inds, feature]
        stat, p_value = mannwhitneyu(group1, group2)
        results.append(p_value)
    results_ndarray = np.array(results)
    return results_ndarray

def rank_relieff(ranking, df_train, y_train, n: int, N_top: int, k: int, strategy: str='mean'):
    """
       This function implements ranking feature selection using ReliefF algorithm
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature":list(df_train.columns)})
    scores.set_index("feature", inplace = True)
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds = df_train.iloc[train_idx]
            df_test_fold = df_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            selector = ReliefF(n_neighbors = 10)
            selector.fit(df_train_folds.values, y_train_fold.values)
            importances = pd.Series(selector.feature_importances_, index = ranking.index)
            top_n_features = importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top,"ReliefF", strategy)


def rank_utest(ranking: pd.DataFrame, df_train: pd.DataFrame, y_train: pd.Series, n: int,
               N_top: int, k: int, strategy: str='mean'):
    """
       This function implements a Mann Whitney U-test based ranking feature selection
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature": list(df_train.columns)})
    scores.set_index("feature", inplace = True)
    alpha = 0.05
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds = df_train.iloc[train_idx]
            df_test_fold = df_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            feature_p_values = u_test(df_train_folds, y_train_fold)
            _, p_vals_adjusted, _, _ = multipletests(pvals = feature_p_values, alpha = alpha, method = 'fdr_bh')
            importances = pd.Series(1 - p_vals_adjusted, index = ranking.index)
            top_n_features = importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top, "Utest", strategy)


def rank_randomforest(ranking, df_train, y_train, n: int, N_top: int, k: int, strategy: str='mean'):
    """
       This function implements ranking feature selection via embedded approach using Random Forest Classifiers
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature": list(df_train.columns)})
    scores.set_index("feature", inplace = True)
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds = df_train.iloc[train_idx]
            df_test_fold = df_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            selector = RandomForestClassifier(n_jobs = -1)
            selector.fit(df_train_folds, y_train_fold)
            importances = pd.Series(selector.feature_importances_, index = ranking.index)
            top_n_features = importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top, "RF", strategy)


def rank_logisticregression(ranking, df_train, y_train, n: int, N_top: int, k: int, strategy: str='mean'):
    """
       This function implements ranking feature selection via embedded approach using Logistic Regression algorithm
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature": list(df_train.columns)})
    scores.set_index("feature", inplace = True)
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds=df_train.iloc[train_idx]
            df_test_fold=df_train.iloc[test_idx]
            y_train_fold=y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            selector = LogisticRegression(penalty = 'l1', solver = 'liblinear', random_state = 50, max_iter = 1000)
            selector.fit(df_train_folds, y_train_fold)
            importances = pd.Series(np.abs(selector.coef_[0]), index = ranking.index)
            top_n_features = importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top, "LogReg", strategy)


def rank_xgboost(ranking, df_train, y_train, n: int, N_top: int, k: int, strategy: str='mean'):
    """
       This function implements ranking feature selection via embedded approach using XGBClassifier
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature": list(df_train.columns)})
    scores.set_index("feature", inplace = True)
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds = df_train.iloc[train_idx]
            df_test_fold = df_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            selector = XGBClassifier(n_estimators = 300, learning_rate = 0.05)
            selector.fit(df_train_folds, y_train_fold)
            xgb_importances = selector.get_booster().get_score(importance_type = 'gain')
            full_feature_importances = pd.Series(0, index = scores.index)
            map_features = {f"f{i}": col for i, col in enumerate(scores.index)}
            for key in xgb_importances.keys():
                if key in map_features.keys():
                    full_feature_importances[map_features[key]] = xgb_importances[key]
            # print(full_feature_importances.sort_values(ascending=False).head(10))
            top_n_features = full_feature_importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top, "XGB", strategy)

def rank_fisher(ranking, df_train, y_train, n: int, N_top: int, k: int, strategy: str='mean'):
    """
       This function implements ranking feature selection using Fisher Score algorithm
       combined with repeated cross validation for stability.

       Args:
           ranking (pd.DataFrame): Ranking generated based on the features contained in the input dataset.
           This object is updated in-place with new feature ranking columns calculated according
           to above-mentioned selection method.
           df_train (pd.DataFrame): Training dataset.
           y_train (pd.Series): Vector of class labels for training dataset.
           n (int): Number of cross-validation repetitions.
           N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
           method.
           k (int): Number of folds to split the dataset into within cross-validation.
           strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

       Returns:
           None.
    """
    counter = 0
    scores = pd.DataFrame({"feature": list(df_train.columns)})
    scores.set_index("feature", inplace=True)
    for i in range(n):
        skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = 50)
        for train_idx, test_idx in skf.split(df_train, y_train):
            df_train_folds = df_train.iloc[train_idx]
            df_test_fold = df_train.iloc[test_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_test_fold = y_train.iloc[test_idx]
            importances = pd.Series(fisher_score(df_train_folds.values, y_train_fold.values), index = ranking.index)
            top_n_features = importances.sort_values(ascending = False).head(N_top)
            scores[f'Ranking {counter}'] = top_n_features
            counter += 1
    scores_copy = scores.copy()
    inside_aggregation(scores_copy, ranking, scores, N_top, "Fisher", strategy)



def initialize_ranking(df_train: pd.DataFrame):
    """
        This function initializes feature ranking for ensemble feature selection based on given input dataset.

        Args:
            df_train (pd.DataFrame): Input dataset.
            This object contains features needed to index through the output ranking.

        Returns:
            ranking (pd.DataFrame): Initialized ranking.
            This object is successively updated with new rankings coming from base feature selection methods.
    """
    ranking = pd.DataFrame({"feature": list(df_train.columns)})
    ranking.set_index("feature", inplace = True)
    return ranking

def inside_aggregation(scores_copy: pd.DataFrame, ranking: pd.DataFrame,
                       scores: pd.DataFrame, N_top: int, method: str, strategy_for_soft: str):
    """
        This function performs inside rankings aggregation in each base method specified for feature selection.

        Args:
            scores_copy (pd.DataFrame): Copy of DataFrame with feature relevances calculated by given method.
            ranking (pd.DataFrame): Target ranking to be updated with newly calculated base rankings
            derived from individual feature selection methods.
            scores (pd.DataFrame): Original DataFrame with feature relevances calculated by given method.
            N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
            method.
            method (str): Name of the selection method calling the above mentioned function.
            strategy_for_soft (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

        Returns:
            None.

    """
    rankings_per_iter = [col for col in scores.columns if "Ranking" in col]
    strategies = {"mean": lambda scores_copy: scores_copy[rankings_per_iter].mean(axis = 1),
                  "median": lambda scores_copy: scores_copy[rankings_per_iter].median(axis = 1),
                  "min": lambda scores_copy: scores_copy[rankings_per_iter].min(axis = 1),
                  "max": lambda scores_copy: scores_copy[rankings_per_iter].max(axis = 1)}
    for column in rankings_per_iter:
        indices = scores[column].sort_values(ascending = False)[:N_top].index
        scores.loc[indices, column] = 1
        scores.loc[~scores.index.isin(indices), column] = 0
    hard_vote_col = scores[rankings_per_iter].sum(axis = 1)
    ranking[f'Scores {method} Hard'] = hard_vote_col
    if strategy_for_soft.lower().strip() in strategies.keys():
        soft_vote_col = strategies[strategy_for_soft.lower().strip()](scores_copy)
    else:
        raise ValueError("Invalid strategy named " + strategy_for_soft + " was entered. Try again.")
    ranking[f'Scores {method} Soft'] = soft_vote_col


def hard_voting(ranking: pd.DataFrame):
    """
        This function performs external aggregation of the rankings obtained by ensemble feature selection
        using Hard Voting approach. As a result the input ensemble ranking is updated with a column assigning
        the number of votes received to each feature.

         Args:
             ranking (pd.DataFrame): The output ranking DataFrame containing all rankings collected by
              internal feature selection methods.

         Returns:
             None.
    """
    hardvoting_cols = [col for col in ranking.columns if "Hard" in col]
    for col in hardvoting_cols:
        ranking[col] = ranking[col].apply(lambda x: 1 if x >= 1 else 0)
    ranking["Hard Voting"] = ranking[hardvoting_cols].sum(axis = 1)



def soft_voting(ranking: pd.DataFrame, strategy: str):
    """
        This function performs external aggregation of the rankings obtained by ensemble feature selection
        using Soft Voting approach. As a result the input ensemble ranking is updated with a column assigning
        the final aggregated score received to each feature.

         Args:
             ranking (pd.DataFrame): The output ranking DataFrame containing all rankings collected by
              internal feature selection methods.

         Returns:
             None.
    """
    softvoting_cols = [col for col in ranking.columns if "Soft" in col]
    available_strategies = {"mean": lambda ranking: ranking[softvoting_cols].mean(axis = 1),
                            "median": lambda ranking: ranking[softvoting_cols].median(axis = 1),
                            "min": lambda ranking: ranking[softvoting_cols].min(axis = 1),
                            "max": lambda ranking: ranking[softvoting_cols].max(axis = 1)}
    if strategy.lower().strip() not in available_strategies:
        raise ValueError("Unsupported Soft Voting strategy named " + strategy + ". Try again.")
    votes = available_strategies[strategy.lower().strip()](ranking)
    ranking["Soft Voting"] = votes

def feature_selection(methods: list, voting_strategy: str,
                      threshold: int | float, outer_strategy_soft: str=None, return_ranking: bool=False, **kwargs):
    """
        This function performs ensemble feature selection using methods from the given input list.

        Args:
            methods (list): List of methods to include in ensemble feature selection.
            voting_strategy (str): Name of voting strategy used for external aggregation of obtained rankings.
            threshold (int or float): Threshold that determines whether a feature will be included in the resulting selected subset.
            -If voting_strategy == 'hard', an integer specifying the minimum amount of votes required for a feature to be selected,
            - If voting_strategy == 'soft' a float specifying the minimum aggregated score required for a feature to be selected.
            outer_strategy_soft (str):  Strategy of external rankings aggregation only applicable when Soft Voting method
            is chosen (e.g., 'mean', 'median').
            **kwargs (dict): Dictionary of input arguments of the base selection methods, common to each one of them:
                -df_train (pd.DataFrame): Training dataset.
                -y_train (pd.Series): Vector of class labels for training dataset.
                -n (int): Number of cross-validation repetitions.
                -N_top (int): Number of the most important features needed for internal rankings aggregation using Hard Voting
                method.
                -k (int): Number of folds to split the dataset into within cross-validation.
                -strategy (str): Strategy of internal rankings aggregation using Soft Voting method (e.g., 'mean', 'median').

        Returns:
            selected_features (list): List of selected features.
    """
    df_train = kwargs.get("df_train")
    if not isinstance(df_train, pd.DataFrame) or df_train.empty:
        raise ValueError("Dataset must be given as a DataFrame and must not be empty. Try again.")
    ranking = initialize_ranking(df_train)
    available_methods={"relieff": lambda ranking, **kwargs: rank_relieff(ranking, **kwargs),
                       "utest": lambda ranking, **kwargs: rank_utest(ranking, **kwargs),
                       "randomforest": lambda ranking, **kwargs: rank_randomforest(ranking, **kwargs),
                       "logisticregression": lambda ranking, **kwargs: rank_logisticregression(ranking, **kwargs),
                       "xgboost": lambda ranking, **kwargs: rank_xgboost(ranking, **kwargs),
                       "fisher": lambda ranking, **kwargs: rank_fisher(ranking, **kwargs)}
    available_soft_strategies = ["mean", "median", "max", "min", "sum"]
    for method in methods:
        method_normalized = method.lower().replace(" ", "").replace("_", "").replace("-", "").strip()
        if method_normalized in list(available_methods.keys()):
            try:
                method_normalized = available_methods[method_normalized](ranking, **kwargs)
            except (TypeError, KeyError) as e:
                raise ValueError("Invalid dictionary content: " + str(e))
        else:
            raise ValueError("Unsupported method named " + method_normalized + " appeared on the list. Try again.")
    if voting_strategy.lower() == "hard" and isinstance(threshold, int):
        hard_voting(ranking)
        filtered_scores = ranking[ranking["Hard Voting"] >= threshold]
    elif voting_strategy.lower() == "soft" and isinstance(threshold, float):
        if outer_strategy_soft.lower() in available_soft_strategies:
            soft_voting(ranking, outer_strategy_soft)
            filtered_scores = ranking[ranking["Soft Voting"] >= threshold]
        else:
            raise ValueError("Unsupported Soft Voting strategy named " + outer_strategy_soft + ". Try again.")
    else:
        raise ValueError("To run feature selection strategies such as hard or soft voting must be entered "
                         "with their corresponding threshold parameter type. Try again.")
    # print("Filtered scores: ", filtered_scores)
    selected_features = filtered_scores.index.to_list()
    if return_ranking:
        return selected_features, ranking
    else:
        return selected_features

















