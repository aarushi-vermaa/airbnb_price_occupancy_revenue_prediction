import os
import pickle
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from utils import classify_columns, naive_predictions

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
# Price prediction
price_df = pd.read_csv("Data/Preprocessed_data/hawaii_reg.csv")
cols_to_drop = ['desc_1', 'desc_2', 'desc_3',
                'desc_4', 'desc_5', 'n_1',
                'n_2', 'n_3']
price_df.drop(cols_to_drop, axis=1, inplace=True)


def split_listing_ids(train_size: float = 0.8):
    """
    To avoid spillovers between listings, we need to
    split the data so that one listing cannot be in the validation
    and train sets
    :param train_size:
    :return:
    """
    np.random.seed(1234)
    listing_ids = price_df.id.unique()
    np.random.shuffle(listing_ids)
    nr_listings = len(listing_ids)
    t_size = int(nr_listings * train_size)
    train_ids = listing_ids[:t_size]
    val_ids = listing_ids[t_size:]
    return train_ids, val_ids


def split_listings_by_ids():
    train_ids, val_ids = split_listing_ids()
    train_df = price_df[price_df.id.isin(train_ids)].copy()
    val_df = price_df[price_df.id.isin(val_ids)].copy()
    train_df.drop("id", axis=1, inplace=True)
    val_df.drop("id", axis=1, inplace=True)
    return train_df, val_df


train_df, val_df = split_listings_by_ids()

x_train = train_df.iloc[:, :-2]
y_train_mean = train_df.iloc[:, -2]
y_train_std = train_df.iloc[:, -1]

x_val = val_df.iloc[:, :-2]
y_val_mean = val_df.iloc[:, -2]
y_val_std = val_df.iloc[:, -1]

x_train, x_val = classify_columns(x_train, x_val)

results_df = pd.DataFrame(columns=['model', 'rmse', 'mae'])
naive_preds = naive_predictions(train_df)
naive_preds = (pd.merge(val_df[['month', 'log_price_mean']], naive_preds, on='month')
               .drop('month', axis=1).rename({"log_price_mean_x": "true_y",
                                              "log_price_mean_y": "predicted_y"}, axis=1))
results_df.loc[0, 'model'] = 'naive predictions'
results_df.loc[0, 'rmse'] = (np.sqrt(mean_squared_error(naive_preds.true_y,
                                                        naive_preds.predicted_y)))
results_df.loc[0, 'mae'] = mean_absolute_error(naive_preds.true_y,
                                               naive_preds.predicted_y)

default_models = [(xgb.XGBRegressor(random_state=1234), 'xgboost_reg'),
                  (lgb.LGBMRegressor(random_state=1234), 'lgbm_reg'),
                  (RandomForestRegressor(random_state=1234), 'rf_reg')]

for idx, (model, model_name) in tqdm(enumerate(default_models, 1)):
    reg = model
    reg.fit(x_train, y_train_mean)
    preds = reg.predict(x_val)
    results_df.loc[idx, 'model'] = model_name + " default"
    results_df.loc[idx, 'rmse'] = (np.sqrt(mean_squared_error(y_val_mean,
                                                              preds)))
    results_df.loc[idx, 'mae'] = mean_absolute_error(y_val_mean, preds)

# XGBoost optimization
lgbm_reg = lgb.LGBMRegressor(random_state=1234)
mse = make_scorer(mean_squared_error,
                  greater_is_better=False,
                  needs_threshold=False)

search_spaces = {'max_depth': Integer(3, 40),
                 'learning_rate': Real(0.01, 0.3, 'log-uniform'),
                 'n_estimators': Integer(100, 1000)}

opt_lgbm = BayesSearchCV(lgbm_reg,
                         search_spaces,
                         scoring=mse,
                         cv=3,
                         n_iter=3,
                         return_train_score=True,
                         refit=True,
                         optimizer_kwargs={'base_estimator': 'GP'},
                         random_state=1234)
opt_lgbm.fit(x_train, y_train_mean)
opt_preds = opt_lgbm.predict(x_val)
results_df.loc[4, 'model'] = "lgbm optimized"
results_df.loc[4, 'rmse'] = (np.sqrt(mean_squared_error(y_val_mean, opt_preds)))
results_df.loc[4, 'mae'] = mean_absolute_error(y_val_mean, opt_preds)
results_df.to_csv("Data/reg_model_val_results.csv", index=False)

# Train on all training data
best_params = dict(opt_lgbm.best_params_)
X = pd.concat([x_train, x_val], axis=0)
y = pd.concat([y_train_mean, y_val_mean], axis=0)
opt_lgbm = lgb.LGBMRegressor(random_state=1234, **best_params)
opt_lgbm.fit(X, y)
opt_lgbm.booster_.save_model("Models/lgbm_reg_opt.txt")

xgb_reg = xgb.XGBRegressor(random_state=1234)
xgb_reg.fit(X, y)
xgb_reg.save_model("Models/xgb_reg.json")
