import Preprocessing
import Building
import FeatureSelection
import Visualisation

excel_file = r"C:\Users\Dell\Desktop\data.train.SM.control.untreat.xlsx"
df_features, target_col = Preprocessing.read_data(excel_file, 'class', True)
df_preprocessed = Preprocessing.preprocess(df_features, target_col, deal_with_nan_val = False,
                                           scale_data = True, code_target_= False)
df_train, df_test, y_train, y_test = Preprocessing.split(df_preprocessed, target_col, 0.3,
                                                         50, True, target_col)

sel_feat, ranking = FeatureSelection.feature_selection(["relieff", "utest", "randomforest",
                                                        "fisher", "logisticregression", "xgboost"],
                           "hard", 3, None, True,
                                                       **{"df_train": df_train,
                                                        "y_train": y_train, "n": 5, "N_top": 100,
                                                          "k": 5, "strategy": "mean"})
print(sel_feat)
df_train_sel = df_train[sel_feat]

tuning_dict = Building.crossvalidate_and_tune("svm", df_train_sel, y_train,
                                              True, True)
best_params = tuning_dict["best_params"]
print(best_params)
best_score = tuning_dict["best_score"]

model = Building.build_and_train("svm", df_train_sel, y_train, **best_params)

df_test_sel = df_test[sel_feat]
final_dict = Building.test_and_evaluate(["accuracy", "f1", "recall", "mcc", "rocauc", "mse"],
                                        model, df_test_sel, y_test, True)
print(final_dict["scoring"])

cv_results = tuning_dict["cv_results"]
path_to_save = r"C:\Users\Dell\PycharmProjects\enSMble"
Visualisation.feature_importances_plot(model.feature_importances_, list(df_train_sel.columns),
                                       path_to_save = path_to_save)
Visualisation.feature_corr_heatmap(df_train_sel, path_to_save = path_to_save)
Visualisation.hyperparams_validation_curve(cv_results, ["kernel", "C", "gamma"],
                                           path_to_save = path_to_save)
Visualisation.class_prediction_error_plot(y_test, final_dict["y_pred"],
                                          path_to_save = path_to_save)
Visualisation.class_imbalance_barplot(target_col, path_to_save = path_to_save)
Visualisation.class_imbalance_scatterplot(df_train_sel, y_train,
                                          2, path_to_save = path_to_save)
Visualisation.class_imbalance_scatterplot(df_train_sel, y_train,
                                          3, path_to_save = path_to_save)


