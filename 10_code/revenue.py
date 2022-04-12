import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from utils import classify_columns

# Price
original_data = pd.read_csv("Data/Train/listings.csv.gz", compression='gzip')
price_df = pd.read_csv("Data/Preprocessed_data/hawaii_reg.csv")
price_df_id = price_df.copy()
cols_to_drop = ['desc_1', 'desc_2', 'desc_3',
                'desc_4', 'desc_5', 'n_1',
                'n_2', 'n_3', 'id']
price_df.drop(cols_to_drop, axis=1, inplace=True)

availability_df = pd.read_csv("Data/Preprocessed_data/hawaii_cat.csv")

price_df, _ = classify_columns(price_df, price_df)
price_df_id, _ = classify_columns(price_df_id, price_df_id)
price_x = price_df.iloc[:, :-2]
price_y = price_df.iloc[:, -2]

xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("Models/xgb_reg.json")

price_pred = xgb_reg.predict(price_x)
price_df_id['predicted_price'] = price_pred

# Availability
avail_y = availability_df.iloc[:, -1]
avail_x = availability_df.iloc[:, :-1]
avail_x, _ = classify_columns(avail_x, avail_x)

xgb_cls = xgb.XGBClassifier()
xgb_cls.load_model("Models/xgb_cls.json")
avail_pred = xgb_cls.predict(avail_x)

availability_df['availability_predicted'] = avail_pred
availability_df['id'] = original_data[['id']]

price_df_id['id'] = price_df_id['id'].astype(int)

merged_df = pd.merge(price_df_id,
                     availability_df[['id', 'target', 'availability_predicted']],
                     on='id')
# Analysis
conditions = [
    merged_df.availability_predicted == 0,
    merged_df.availability_predicted == 1,
    merged_df.availability_predicted == 2
]

lower_choices = [
    0.7,
    0.3,
    0.0
]

upper_choices = [
    1.0,
    0.69,
    0.29
]

merged_df['lower_t'] = np.select(conditions, lower_choices)
merged_df['upper_t'] = np.select(conditions, upper_choices)
merged_df['lower_revenue'] = merged_df.lower_t * merged_df.predicted_price
merged_df['upper_revenue'] = merged_df.upper_t * merged_df.predicted_price

merged_df['lower_revenue'] = np.exp(merged_df.lower_revenue) * 365
merged_df['upper_revenue'] = np.exp(merged_df.upper_revenue) * 365

example = merged_df.query('id == 13688')
merged_df.to_csv("Data/Preprocessed_data/revenue.csv", index=False)

with_ac = merged_df.query("has_ac == 1 & accommodates == 6 & minimum_nights == 3 & bedrooms == 2")
