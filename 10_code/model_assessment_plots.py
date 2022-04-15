import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ROCAUC
from utils import classify_columns

######## Regression Model Plots
# read in the revenue file we generated 
data = pd.read_csv("../20_intermediate_files/revenue.csv")
# generate residuals for our regression model (XGBoost Regressor)
data['residuals'] = data.log_price_mean - data.predicted_price
data['exp_pred'] = np.exp(data.predicted_price)
data['exp_price'] = np.exp(data.log_price_mean)
data['exp_resids'] = data.exp_pred - data.exp_price
data['abs_exp_resids'] = np.abs(data.exp_resids)
data['abs_resids'] = np.abs(data.residuals)


# plot residual plot
sns.set_style('dark')
sns.set(rc={'figure.figsize': (14, 12)}, font_scale=1.2)
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Actual vs Predicted log price')
ax[1].set_title('Residual plot')
sns.histplot(data, x='log_price_mean',
             bins=50, alpha=0.1, color='blue',
             label='actual log price', kde=True, ax=ax[0]).set(xlabel='Log price')
sns.histplot(data, x='predicted_price',
             bins=50, alpha=0.1, color='orange',
             label='predicted log price', kde=True, ax=ax[0])
sns.histplot(data, x='residuals', bins=50, ax=ax[1]).set(xlabel='Residuals')
ax[0].legend()
#plt.savefig("../30_results/Plots/residuals.png", dpi=600)
plt.show()

# plot residual plot distribution (log scale)
fig, ax = plt.subplots(1, 2)
ax[0].set_title('Actual vs Predicted price')
ax[1].set_title('Residual plot')
sns.histplot(data, x='exp_price',
             bins=50, alpha=0.1, color='blue',
             label='actual price', kde=True, ax=ax[0]).set(xlabel='Price')
sns.histplot(data, x='exp_pred',
             bins=50, alpha=0.1, color='orange',
             label='predicted price', kde=True, ax=ax[0])
sns.histplot(data, x='exp_resids', bins=50, ax=ax[1]).set(xlabel='Residuals')
ax[0].legend()

plt.savefig("30_results/Plots/residuals_exp.png", dpi=600)
plt.show()

# scatter plot of the actual price v.s. our predicted price
sns.scatterplot(x='log_price_mean',
                y='predicted_price',
                data=data).set(title='Predicted vs Actual price')
plt.savefig("../30_results/Plots/price_scatter.png", dpi=600)
# plt.show()

# plotting the relationship between residual and log price 
sns.relplot(x="log_price_mean", y="residuals", ci=None, kind="line", data=data).set(title='Predicted vs Actual price')
plt.show()

# Run the hawaii data regression 
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("../30_results/Models/xgb_reg.json")
c = pd.read_csv("../00_setup/Data/Preprocessed_data/hawaii_reg.csv")
c.drop(['desc_1', 'desc_2', 'desc_3',
        'desc_4', 'desc_5', 'n_1',
        'n_2', 'n_3', 'id', 'log_price_mean', 'log_price_std'],
       axis=1, inplace=True)
c, _ = classify_columns(c, c)

# plot the feature importance plot (sorted by importance)
sorted_idx = xgb_reg.feature_importances_.argsort()
plt.title("Feature Importance", fontsize=15)
plt.barh(c.columns[sorted_idx], xgb_reg.feature_importances_[sorted_idx])
plt.savefig("../30_results/Plots/feature_importance_reg.png", bbox_inches='tight', dpi=600)
plt.show()

########## Classification model
xgb_cls = xgb.XGBClassifier()
xgb_cls.load_model("30_results/Models/xgb_cls.json")

# read in the preprocessed data of the classification model 
availability_df = pd.read_csv("../00_setup/Data/Preprocessed_data/hawaii_cat.csv")

train_df, val_df = train_test_split(availability_df,
                                    train_size=0.8,
                                    random_state=1234,
                                    stratify=availability_df.target)

x_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

x_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1]

x_train, x_val = classify_columns(x_train, x_val)

# Plot ROC AUC curve for validation set for XGBoost Classification
visualizer1 = ROCAUC(xgb_cls, classes=["0", "1", "2"])
visualizer1.fit(x_train, y_train)
visualizer1.score(x_val, y_val)
visualizer1.show("../30_results/Plots/roc_curve_val.png", dpi=600)

# read in the preprocessed test data 
broward_av = pd.read_csv("../00_setup/Data/Preprocessed_data/broward_cat.csv")
crete_av = pd.read_csv("../00_setup/Data/Preprocessed_datacrete_cat.csv")

broward_c_y = broward_av.iloc[:, -1]
broward_c_x = broward_av.iloc[:, :-1]
broward_c_x, _ = classify_columns(broward_c_x, broward_c_x)

crete_c_y = crete_av.iloc[:, -1]
crete_c_x = crete_av.iloc[:, :-1]
crete_c_x, _ = classify_columns(crete_c_x, crete_c_x)

# Plot ROC AUC plot for Broward county using XGBoost Classification
visualizer2 = ROCAUC(xgb_cls, classes=["0", "1", "2"])
visualizer2.fit(broward_c_x, broward_c_y)
visualizer2.score(broward_c_x, broward_c_y)
visualizer2.show("../30_results/Plots/roc_curve_test_broward.png", dpi=600)

# Plot ROC AUC plot for Crete using XGBoost Classification
visualizer3 = ROCAUC(xgb_cls, classes=["0", "1", "2"])
visualizer3.fit(crete_c_x, crete_c_y)
visualizer3.score(crete_c_x, crete_c_y)
visualizer3.show("../30_results/Plots/roc_curve_test_crete.png", dpi=600)

# plot the feature importance plot (sorted by importance)
sorted_idx = xgb_cls.feature_importances_.argsort()
plt.title("Feature Importance", fontsize=15)
plt.barh(crete_c_x.columns[sorted_idx], xgb_cls.feature_importances_[sorted_idx])
plt.savefig("../30_results/Plots/feature_importance_cls.png", bbox_inches='tight', dpi=600)
plt.show()

