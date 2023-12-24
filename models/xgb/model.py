import xgboost
import sklearn

xgb_best_params = {'max_leaf_nodes': 74, 'min_samples_split': 5}
xgb = xgboost.XGBClassifier(**xgb_best_params)
