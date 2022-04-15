import pandas as pd
### Helper functions to import in other files 


def classify_columns(x_train: pd.DataFrame, x_val: pd.DataFrame):
    """
    input: 
    x_train: dataframe for training data
    x_val: dataframe for validation set
    return:
    x_train: processed dataframe for training data
    x_val: processed dataframe for validation data
    """
    categorical_features = [
        'host_response_time', 'room_type', 'host_has_profile_pic',
        'host_identity_verified', 'bathrooms_text', 'bedrooms',
        'beds', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'instant_bookable', 'has_wifi',
        'has_pool', 'has_kitchen', 'has_washer', 'has_dryer',
        'has_ac', 'has_self_checkin', 'has_workspace', 'has_pet_allowed',
        'has_free_parking'
    ]
    numerical_features = [col for col in x_train.columns if col not in categorical_features]
    feature_names = categorical_features + numerical_features
    feature_types = ['categorical' for _ in range(len(categorical_features))] + \
                    ['continuous' for _ in range(len(numerical_features))]
    # convert the text version columns (categorical columns) to int
    x_train[categorical_features] = x_train[categorical_features].astype(int)
    x_val[categorical_features] = x_val[categorical_features].astype(int)
    # the rest of the columns stay the same 
    x_train = x_train[feature_names]
    x_val = x_val[feature_names]
    # convert the numerical columns to float
    x_train[numerical_features] = x_train[numerical_features].astype(float)
    x_val[numerical_features] = x_val[numerical_features].astype(float)
    return x_train, x_val


def naive_predictions(df: pd.DataFrame):
    """
    Generate Baseline model for our availabiity classification 
    Generate prediction using mean of the price
    """
    return df.groupby(['month'], as_index=False)[['log_price_mean']].mean()

