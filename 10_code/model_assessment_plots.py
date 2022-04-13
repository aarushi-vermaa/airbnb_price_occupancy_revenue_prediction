import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import classify_columns

# %%
data = pd.read_csv("20_intermediate_files/revenue.csv")
data['residuals'] = data.log_price_mean - data.predicted_price
data['exp_pred'] = np.exp(data.predicted_price)
data['exp_price'] = np.exp(data.log_price_mean)
data['exp_resids'] = data.exp_pred - data.exp_price
data['abs_exp_resids'] = np.abs(data.exp_resids)
data['abs_resids'] = np.abs(data.residuals)


# %%
sns.set_style('dark')
sns.set(rc={'figure.figsize': (12, 8)})
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
plt.savefig("30_results/Plots/residuals.png", dpi=600)
plt.show()

# %%
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

# %%
sns.scatterplot(x='log_price_mean',
                y='predicted_price',
                data=data).set(title='Predicted vs Actual price')
plt.savefig("30_results/Plots/price_scatter.png", dpi=600)
# plt.show()

# %%
sns.relplot(x="log_price_mean", y="residuals", ci=None, kind="line", data=data).set(title='Predicted vs Actual price')
plt.show()

# %%
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("Models/xgb_reg.json")
c = pd.read_csv("../Data/Preprocessed_data/hawaii_reg.csv")
c.drop(['desc_1', 'desc_2', 'desc_3',
        'desc_4', 'desc_5', 'n_1',
        'n_2', 'n_3', 'id', 'log_price_mean', 'log_price_std'],
       axis=1, inplace=True)
c, _ = classify_columns(c, c)

# %%
sorted_idx = xgb_reg.feature_importances_.argsort()
plt.barh(c.columns[sorted_idx], xgb_reg.feature_importances_[sorted_idx])
plt.show()
