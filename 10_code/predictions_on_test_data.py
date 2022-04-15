import numpy as np
import pandas as pd
import xgboost as xgb

from utils import classify_columns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report

# Load the data of testing data 
broward_price = pd.read_csv("../00_setup/Data/Preprocessed_data/broward_reg.csv")
broward_av = pd.read_csv("../00_setup/Data/Preprocessed_data/broward_cat.csv")

crete_price = pd.read_csv("../00_setup/Data/Preprocessed_data/crete_reg.csv")
crete_av = pd.read_csv("../00_setup/Data/Preprocessed_data/crete_cat.csv")

# remove irrelevant columns
cols_to_drop = ['desc_1', 'desc_2', 'desc_3',
                'desc_4', 'desc_5', 'n_1',
                'n_2', 'n_3', 'id']

broward_price.drop(cols_to_drop, axis=1, inplace=True)
crete_price.drop(cols_to_drop, axis=1, inplace=True)

broward_p_y = broward_price.iloc[:, -2]
broward_p_x = broward_price.iloc[:, :-2]
broward_p_x, _ = classify_columns(broward_p_x, broward_p_x)

crete_p_y = crete_price.iloc[:, -2]
crete_p_x = crete_price.iloc[:, :-2]
crete_p_x, _ = classify_columns(crete_p_x, crete_p_x)

broward_c_y = broward_av.iloc[:, -1]
broward_c_x = broward_av.iloc[:, :-1]
broward_c_x, _ = classify_columns(broward_c_x, broward_c_x)

crete_c_y = crete_av.iloc[:, -1]
crete_c_x = crete_av.iloc[:, :-1]
crete_c_x, _ = classify_columns(crete_c_x, crete_c_x)

###### Regression model
xgb_reg = xgb.XGBRegressor()
# load the model we have trained on Hawaii
xgb_reg.load_model("../30_results/Models/xgb_reg.json")

# apply prediction to the 2 test cities
xgb_reg_b_pred = xgb_reg.predict(broward_p_x)
xgb_reg_c_pred = xgb_reg.predict(crete_p_x)

###### Classification model
xgb_cls = xgb.XGBClassifier()
# load the model we have trained on Hawaii
xgb_cls.load_model("../30_results/Models/xgb_cls.json")

# apply prediction to the 2 test cities
xgb_cls_b_pred = xgb_cls.predict(broward_c_x)
xgb_cls_c_pred = xgb_cls.predict(crete_c_x)

#### Metrics
##### Regression
# save the result to model with city, model name and the metrics we're using: RMSE and MAE
# save broward county RMSE and MAE
results_df = pd.DataFrame(columns=['city', 'model', 'rmse', 'mae'])
results_df.loc[0, "city"] = 'Broward'
results_df.loc[0, "model"] = 'xgboost_reg default'
results_df.loc[0, "rmse"] = np.sqrt(mean_squared_error(broward_p_y, xgb_reg_b_pred))
results_df.loc[0, "mae"] = mean_absolute_error(broward_p_y, xgb_reg_b_pred)
# save Crete RMSE and MAE
results_df.loc[1, "city"] = 'Crete'
results_df.loc[1, "model"] = 'xgboost_reg default'
results_df.loc[1, "rmse"] = np.sqrt(mean_squared_error(crete_p_y, xgb_reg_c_pred))
results_df.loc[1, "mae"] = mean_absolute_error(crete_p_y, xgb_reg_c_pred)
results_df.to_csv("../30_results/Model Results/reg_model_test_results.csv", index=False)


##### Classification

# save the result to model with city, model name and the metrics we're using: Precision, Recall, F1, support
def cls_report(true_y, pred_y, city, model):

    r_df = (
        pd.DataFrame(classification_report(true_y,
                                           pred_y,
                                           output_dict=True))
            .transpose()
            .reset_index()
    )
    r_df['model'] = model
    r_df['city'] = city
    col_order = ['city', 'model', 'index', 'precision',
                 'recall', 'f1-score', 'support']
    r_df = r_df[col_order]
    return r_df

# save crete PR, F1, support
br_cls_report = cls_report(broward_c_y, xgb_cls_b_pred,
                           'Broward', 'xgboost_cls default')
cr_cls_report = cls_report(crete_c_y, xgb_cls_c_pred,
                           'Crete', 'xgboost_cls default')
cls_report = pd.concat([br_cls_report, cr_cls_report], axis=0)
cls_report.to_csv("../30_results/Model Results/clf_model_test_results.csv", index=False)

# function to generate XGBoost classifciation confusion matrix values
def conf_m(true_y, pred_y, city, model='xgboost_cls default'):

    cm = confusion_matrix(true_y, pred_y)
    cm = pd.DataFrame(cm)
    cm['model'] = model
    cm['city'] = city
    return cm

# save broward county confusion matrix
br_conf = conf_m(broward_c_y, xgb_cls_b_pred, 'Broward')
# save crete confusion matrix
cr_conf = conf_m(crete_c_y, xgb_cls_c_pred, 'Crete')
# put them together 
conf_m = pd.concat([br_conf, cr_conf], axis=0)
conf_m.to_csv("../30_results/Model Results/clf_test_conf_matrix.csv", index=False)
