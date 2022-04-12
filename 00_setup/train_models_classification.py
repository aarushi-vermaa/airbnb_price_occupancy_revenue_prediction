import pickle
import numpy as np
import pandas as pd
import warnings

import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import classify_columns

warnings.filterwarnings('ignore')
availability_df = pd.read_csv("Data/Preprocessed_data/hawaii_cat.csv")

train_df, val_df = train_test_split(availability_df,
                                    train_size=0.8,
                                    random_state=1234,
                                    stratify=availability_df.target)

x_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

x_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1]

x_train, x_val = classify_columns(x_train, x_val)

random_guess = y_val.copy().to_numpy()
np.random.seed(1234)
np.random.shuffle(random_guess)

results_df = (
    pd.DataFrame(classification_report(y_val,
                                       random_guess,
                                       output_dict=True))
        .transpose()
        .reset_index()
)
results_df['model'] = 'random guess'
col_order = ['model', 'index', 'precision', 'recall', 'f1-score', 'support']
results_df = results_df[col_order]
conf_m_df = pd.DataFrame(confusion_matrix(y_val, random_guess))
conf_m_df['model'] = 'random guess'

default_models = [(xgb.XGBClassifier(random_state=1234), 'xgboost_clf'),
                  (lgb.LGBMClassifier(random_state=1234), 'lgbm_clf'),
                  (RandomForestClassifier(random_state=1234), 'rf_clf')]

temp1 = [results_df]
temp2 = [conf_m_df]

for model, model_name in tqdm(default_models):

    clf = model
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)
    cm = confusion_matrix(y_val, preds)
    cm = pd.DataFrame(cm)
    cm['model'] = model_name + " default"
    temp2.append(cm)
    rep = classification_report(y_val, preds, output_dict=True)
    d = pd.DataFrame(rep).transpose().reset_index()
    d['model'] = model_name + " default"
    d = d[col_order]
    temp1.append(d)


# Model optimization
rf_cls = RandomForestClassifier(random_state=1234)

search_spaces = {'max_depth': Categorical([5, 8, 15, 25, 30, None]),
                 'min_samples_split': Integer(1, 100),
                 'min_samples_leaf': Integer(1, 10),
                 'n_estimators': Integer(100, 1200),
                 'max_features': Categorical(['log2', 'sqrt', None])}

opt_rf = BayesSearchCV(rf_cls,
                       search_spaces,
                       scoring='accuracy',
                       cv=3,
                       n_iter=5,
                       return_train_score=True,
                       refit=True,
                       optimizer_kwargs={'base_estimator': 'GP'},
                       random_state=42)

print("Starting optimization... Takes a lot of time...")
opt_rf.fit(x_train, y_train)
preds = opt_rf.predict(x_val)
cm = confusion_matrix(y_val, preds)
cm = pd.DataFrame(cm)
cm['model'] = 'random forest optimized'
temp2.append(cm)
rep = classification_report(y_val, preds, output_dict=True)
d = pd.DataFrame(rep).transpose().reset_index()
d['model'] = 'random forest optimized'
d = d[col_order]
temp1.append(d)

results_df = pd.concat(temp1, axis=0)
results_df.to_csv("Data/clf_model_val_results.csv", index=False)
conf_m_df = pd.concat(temp2, axis=0)
conf_m_df.columns = ['true class 0', 'true class 1', 'true class 2', 'model']
conf_m_df.to_csv("Data/clf_val_conf_matrix.csv", index=False)

best_params = dict(opt_rf.best_params_)
X = pd.concat([x_train, x_val], axis=0)
y = pd.concat([y_train, y_val], axis=0)
rf_cls = RandomForestClassifier(random_state=1234, **best_params)
rf_cls.fit(X, y)

with open("Models/rf_clf.pickle", "wb") as f:
    pickle.dump(rf_cls, f)

xgb_cls = xgb.XGBClassifier(random_state=1234)
xgb_cls.fit(X, y)
xgb_cls.save_model("Models/xgb_cls.json")
