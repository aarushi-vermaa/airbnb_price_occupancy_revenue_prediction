import os
import pickle
import warnings
import numpy as np
import pandas as pd
from re import match
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
warnings.filterwarnings('ignore')

LOCATIONS = [('Data/Train', 'train_h'),
             ('Data/Test_Broward', 'test_b'),
             ('Data/Test_Crete', 'test_c')]
FILES = ['calendar.csv.gz', 'listings.csv.gz']


def load_data():
    """

    :return:
    """
    data = {}
    for loc, name in LOCATIONS:
        for file in FILES:
            filename = "_".join([name, file.split(".")[0]])
            print(f"Loading {filename}")
            data[filename] = pd.read_csv(os.path.join(loc, file),
                                         compression='gzip')
    return data


def calendar_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    print("calendar_preprocessing started")
    data = df.copy()
    data['price'] = data.price.str.split("$").str[1]
    data['adjusted_price'] = data.adjusted_price.str.split("$").str[1]
    data['available'] = data.available.map({"t": 1, "f": 0})
    data['date'] = pd.to_datetime(data.date)
    data['price'] = data['price'].str.replace(',', '')
    data['adjusted_price'] = data['adjusted_price'].str.replace(',', '')
    data[['price', 'adjusted_price']] = (data[['price', 'adjusted_price']]
                                         .astype(float))
    data['month'] = data.date.dt.month
    price_mean_std = (data
                      .groupby(['listing_id', 'month'],
                               as_index=False)[['adjusted_price']]
                      .agg(['mean', 'std'])
                      .reset_index())
    price_mean_std.columns = ['listing_id', 'month', 'price_mean', 'price_std']
    price_mean_std = price_mean_std.query("price_mean > 0.0")
    price_mean_std['log_price_mean'] = np.log(price_mean_std.price_mean)
    price_mean_std['log_price_std'] = np.log(price_mean_std.price_std + 1)
    print("calender data preprocessed")
    return price_mean_std


def impute_len(row):
    try:
        return len(row)
    except:
        return 0


def bathroom_cat(df: pd.DataFrame) -> np.ndarray:
    """

    :param df:
    :return:
    """
    bathroom_condition = [
        df.bathrooms_text < 2,
        df.bathrooms_text < 3,
        df.bathrooms_text < 4,
        df.bathrooms_text >= 4
    ]

    bathroom_choices = [
        1, 2, 3, 4
    ]
    return np.select(bathroom_condition, bathroom_choices, 5)


def bedroom_cat(df: pd.DataFrame) -> np.ndarray:
    """

    :param df:
    :return:
    """
    bedrooms_condition = [
        df.bedrooms < 2,
        df.bedrooms < 3,
        df.bedrooms < 4,
        df.bedrooms < 5,
        df.bedrooms >= 5,

    ]

    bedroom_choices = [
        1, 2, 3, 4, 5
    ]
    return np.select(bedrooms_condition, bedroom_choices, 6)


def bed_cat(df: pd.DataFrame) -> np.ndarray:
    """

    :param df:
    :return:
    """
    beds_condition = [
        df.beds < 2,
        df.beds < 3,
        df.beds < 4,
        df.beds < 5,
        df.beds >= 5,

    ]

    beds_choices = [
        1, 2, 3, 4, 5
    ]
    return np.select(beds_condition, beds_choices, 6)


def ratings_columns(train_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param train_df:
    :return:
    """
    ratings = [col for col in train_df.columns if 'review_scores' in col]
    rating_quantiles = (train_df[ratings]
                        .describe()
                        .filter(['25%', '50%', '75%'], axis=0))
    return rating_quantiles


def to_digits(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df['host_response_rate'] = (
        df['host_response_rate'].str.split("%")
            .str[0].astype(float)
    )

    df['host_acceptance_rate'] = (
        df['host_acceptance_rate'].str.split("%")
            .str[0].astype(float)
    )

    df['bathrooms_text'] = (
        df.bathrooms_text.str.extract(r"(\d+)").astype(float))
    return df


def binary_map(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df['host_has_profile_pic'] = (
        df.host_has_profile_pic.map({"t": 1, "f": 0})
    )

    df['host_identity_verified'] = (
        df.host_identity_verified.map({"t": 1, "f": 0})
    )

    df['instant_bookable'] = (
        df.instant_bookable.map({"t": 1, "f": 0})
    )
    return df


def cat_to_num(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    host_response_time_map = {"within an hour": 1, "within a few hours": 2,
                              "within a day": 3, "a few days or more": 4}
    room_type_map = {"Entire home/apt": 1, "Private room": 2,
                     "Hotel room": 3, "Shared room": 4}

    df['host_response_time'] = (
        df.host_response_time
            .map(host_response_time_map).fillna(5)
    )

    df['room_type'] = (
        df.room_type.map(room_type_map).fillna(5)
    )
    return df


def len_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df['name_len'] = df['name'].str.len()
    df['neighborhood_overview_len'] = (
        df['neighborhood_overview'].str.len()
    )

    df["host_verifications"] = (
        df.host_verifications.apply(lambda x: eval(x))
    )

    df['host_verifications_len'] = (
        df.host_verifications.apply(lambda x: impute_len(x))
    )
    return df


def amenities_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df['amenities_ls'] = (
        df['amenities'].apply(lambda x: eval(x.lower()))
    )
    df['has_wifi'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'wifi' in x else 0)
    )
    df['has_pool'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'pool' in x else 0)
    )
    df['has_kitchen'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'kitchen' in x else 0)
    )
    df['has_washer'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'washer' in x else 0)
    )
    df['has_dryer'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'dryer' in x else 0)
    )
    df['has_ac'] = (
        df['amenities_ls'].apply(lambda x: 1 if 'air conditioning' in x else 0)
    )
    df['has_self_checkin'] = (
        df['amenities_ls']
            .apply(lambda x: 1 if 'self check-in' in x else 0)
    )
    df['has_workspace'] = (
        df['amenities_ls']
            .apply(lambda x: 1 if 'laptop-friendly workspace' in x else 0)
    )
    df['has_pet_allowed'] = (
        df['amenities_ls']
            .apply(lambda x: 1 if 'pets allowed' in x else 0)
    )
    df['has_free_parking'] = (
        df['amenities_ls']
            .apply(lambda x: 1 if bool(filter(lambda v: match(r'free(.*)parking', v), x)) else 0)
    )
    return df


def text_to_features(df: pd.DataFrame, train: bool) -> pd.DataFrame:
    """

    :param df:
    :param train:
    :return:
    """
    if train:
        print("Feature extraction from the text")
        svd_desc = TruncatedSVD(n_components=5, n_iter=8)
        tf_idf_desc = TfidfVectorizer(stop_words='english')

        svd_n = TruncatedSVD(n_components=3, n_iter=8)
        tf_idf_n = TfidfVectorizer(stop_words='english')

        description = tf_idf_desc.fit_transform(df.description.fillna("").to_list())
        description = svd_desc.fit_transform(description)

        neighborhood = tf_idf_n.fit_transform(df.neighborhood_overview.fillna("").to_list())
        neighborhood = svd_n.fit_transform(neighborhood)

        path = "Models"
        if not os.path.exists(path):
            os.mkdir(path)
            print(f'{path} directory created')

        with open("Models/svd_description.pickle", "wb") as f:
            pickle.dump(svd_desc, f)

        with open("Models/svd_neighborhood.pickle", "wb") as f:
            pickle.dump(svd_n, f)

        with open("Models/tfidf_description.pickle", "wb") as f:
            pickle.dump(tf_idf_desc, f)

        with open("Models/tfidf_neighborhood.pickle", "wb") as f:
            pickle.dump(tf_idf_n, f)
        print("Text feature extraction models saved")

    else:
        print("Loading trained feature extraction models")
        with open("Models/svd_description.pickle", "rb") as f:
            svd_desc: TruncatedSVD = pickle.load(f)

        with open("Models/svd_neighborhood.pickle", "rb") as f:
            svd_n: TruncatedSVD = pickle.load(f)

        with open("Models/tfidf_description.pickle", "rb") as f:
            tf_idf_desc: TfidfVectorizer = pickle.load(f)

        with open("Models/tfidf_neighborhood.pickle", "rb") as f:
            tf_idf_n: TfidfVectorizer = pickle.load(f)

        description = tf_idf_desc.transform(df.description.fillna("").to_list())
        description = svd_desc.transform(description)

        neighborhood = tf_idf_n.transform(df.neighborhood_overview.fillna("").to_list())
        neighborhood = svd_n.transform(neighborhood)

    df[['desc_1', 'desc_2', 'desc_3', 'desc_4', 'desc_5']] = description
    df[['n_1', 'n_2', 'n_3']] = neighborhood

    return df


def simple_impute(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df[['host_response_rate', 'host_acceptance_rate']] = df[
        ['host_response_rate', 'host_acceptance_rate']].fillna(0)

    df['host_listings_count'] = df['host_listings_count'].fillna(1)

    df[['host_has_profile_pic', 'host_identity_verified']] = df[
        ['host_has_profile_pic', 'host_identity_verified']].fillna(0)

    df['neighborhood_overview_len'] = (
        df['neighborhood_overview_len'].fillna(0))

    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    return df


def categorize_rating(df: pd.DataFrame, rating_q: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    ratings = [col for col in df.columns if 'review_scores' in col]
    for r in ratings:
        condition_col = rating_q[r]
        condition = [
            df[r] < condition_col['25%'],
            df[r] < condition_col['50%'],
            df[r] < condition_col['75%'],
            df[r] >= condition_col['75%']
        ]
        choices = [1, 2, 3, 4]
        df[r] = np.select(condition, choices, 5)
    return df


def listing_preprocessing(df: pd.DataFrame, train: bool) -> pd.DataFrame:
    """

    :param df:
    :param train:
    :return:
    """
    selected_cols = ['id', 'name', 'description', 'neighborhood_overview',
                     'host_response_time', 'host_verifications',
                     'room_type', 'amenities', 'host_response_rate',
                     'host_acceptance_rate', 'host_listings_count',
                     'host_has_profile_pic', 'host_identity_verified',
                     'accommodates', 'bathrooms_text', 'bedrooms',
                     'beds', 'minimum_nights', 'maximum_nights',
                     'number_of_reviews', 'number_of_reviews_ltm',
                     'review_scores_rating', 'review_scores_accuracy',
                     'review_scores_cleanliness', 'review_scores_checkin',
                     'review_scores_communication', 'review_scores_location',
                     'review_scores_value', 'instant_bookable',
                     'calculated_host_listings_count',
                     'calculated_host_listings_count_entire_homes',
                     'calculated_host_listings_count_private_rooms',
                     'calculated_host_listings_count_shared_rooms',
                     'reviews_per_month', 'availability_365']

    listings_reduced = df[selected_cols].copy()

    # Transform string valued columns into float
    listings_reduced = to_digits(listings_reduced)

    # Map categories into {0, 1}
    listings_reduced = binary_map(listings_reduced)

    # Map string value into numerica categories
    listings_reduced = cat_to_num(listings_reduced)

    # New features
    listings_reduced = len_based_features(listings_reduced)

    # Amenities
    listings_reduced = amenities_one_hot(listings_reduced)

    # Textual features
    listings_reduced = text_to_features(listings_reduced, train)

    listings_reduced['availability_365_rate'] = (
            listings_reduced['availability_365'] / 365
    )

    # Drop unnecessary columns
    listings_reduced.drop(['name', 'description',
                           'neighborhood_overview',
                           'host_verifications', 'amenities',
                           'amenities_ls', 'availability_365'],
                          axis=1,
                          inplace=True)

    # Impute missing values
    listings_reduced = simple_impute(listings_reduced)

    # Bathroom, bedroom, bed
    listings_reduced['bathrooms_text'] = bathroom_cat(listings_reduced)
    listings_reduced['bedrooms'] = bedroom_cat(listings_reduced)
    listings_reduced['beds'] = bed_cat(listings_reduced)

    # Rating columns
    rating_quantiles = ratings_columns(data['train_h_listings'])
    listings_reduced = categorize_rating(listings_reduced, rating_quantiles)

    return listings_reduced


def price_prediction_dataset(listings_df: pd.DataFrame,
                             calendar_df) -> pd.DataFrame:
    """

    :param listings_df:
    :param calendar_df:
    :return:
    """
    l_df = listings_df.copy()
    l_df.drop("availability_365_rate", axis=1, inplace=True)
    merged_df = pd.merge(l_df,
                         calendar_df[['listing_id', 'month',
                                      'log_price_mean', 'log_price_std']],
                         left_on='id',
                         right_on='listing_id')
    merged_df.drop('listing_id', axis=1, inplace=True)

    return merged_df


def availability_prediction_dataset(listings_df: pd.DataFrame,
                                    average_price_df: pd.DataFrame) -> pd.DataFrame:
    """

    :param listings_df:
    :param average_price_df:
    :return:
    """
    merged_df = pd.merge(listings_df, average_price_df,
                         left_on='id', right_on='listing_id')
    merged_df.drop("listing_id", axis=1, inplace=True)
    target_conditions = [
        merged_df.availability_365_rate < 0.30,
        merged_df.availability_365_rate < 0.70,
        merged_df.availability_365_rate >= 0.70
    ]

    target_choices = [0, 1, 2]
    merged_df['target'] = np.select(target_conditions, target_choices)
    merged_df.drop(['availability_365_rate', 'id'], axis=1, inplace=True)
    return merged_df


if __name__ == '__main__':
    data = load_data()

    path = "Data/Preprocessed_data"
    if not os.path.exists(path):
        os.mkdir(path)
        print(f'{path} directory created')

    dfs= [('train_h_listings', 'train_h_calendar',
           True, 'hawaii_reg.csv', 'hawaii_cat.csv'),
          ('test_b_listings', 'test_b_calendar',
           False, 'broward_reg.csv', 'broward_cat.csv'),
          ('test_c_listings', 'test_c_calendar',
           False, 'crete_reg.csv', 'crete_cat.csv')]

    for listing, calendar, t, name_r, name_c in dfs:
        l = listing_preprocessing(data[listing], t)
        c = calendar_preprocessing(data[calendar])
        m = price_prediction_dataset(l, c)
        print(m.shape)
        m.to_csv(os.path.join(path, name_r), index=False)
        print(f'{name_r} saved')

        average_price = (
            data[calendar]
                .groupby(['listing_id'], as_index=False)[['adjusted_price']]
                .mean()
        )
        a = availability_prediction_dataset(l, average_price)
        print(a.shape)
        a.to_csv(os.path.join(path, name_c), index=False)
        print(f'{name_c} saved')
