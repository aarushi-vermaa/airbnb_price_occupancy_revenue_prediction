# Preprocessed data codes description

| #     | Column name               | Column description                                                                                                                  |
|-------|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1     | host_response_time        | "1": within an hour, "2": within a few hours, "3": within a day, "4": a few days or more, "5": missing                              |
| 2     | room_type                 | "1": Entire home/apt,"2": Private room, "3": Hotel room, "4": Shared room, "5": missing                                             |
| 3     | host_has_profile_pic      | "1": yes, "0": no                                                                                                                   |
| 4     | host_identity_verified    | "1": yes, "0": no                                                                                                                   |
| 5     | bathrooms_text            | "1": 1, "2": 2, "3": 3, "4": more than 3, "5": "missing                                                                             |
| 6     | bedrooms                  | "1": 1, "2": 2, "3": 3, "4": 4, "5": more than 4, 6: missing                                                                        |
| 7     | beds                      | "1": 1, "2": 2, "3": 3, "4": 4, "5": more than 4, 6: missing                                                                        |
| 8-14  | review_*                  | quantiles based: "1": [0-25), "2": [25-50), "3": [50-75), "4": [75-100], "missing": 5                                               |
| 15    | instant_bookable          | "1": yes, "0": no                                                                                                                   |
| 16-25 | has_*                     | "1": has *, "0": does not have                                                                                                      |
| 26-39 |                           | original                                                                                                                            |
| 40    | name_len                  | number of symbols in the name of the listing                                                                                        |
| 41    | neighborhood_overview_len | number of symbols in the overview                                                                                                   |
| 42    | host_verifications_len    | number of verification methods (selfie, facebook, etc...)                                                                           |
| 43-47 | desc_*                    | features extracted from the descriptions using Tf-Idf and SVD for dimensionality reduction                                          |
| 48-50 | n_*                       | features extracted from the neighborhood description using Tf-Idf and SVD for dimensionality reduction                              |
| 51    | month                     | 1-jan, 2-feb,...,12-dec                                                                                                             |
| 52    | log_price_mean            | mean price aggregated by month and listing_id from calendar data and log transformed                                                |
| 53    | log_price_std             | std aggregated by month and listing_id from calendar data and log transformed                                                       |
| 54    | predicted_price           | log price predicted by xgboost regression model                                                                                     |
| 55    | target                    | based on availability_365 original column normalized to be in [0, 1] and categorized: "0": [0, 0.3), "1": [0.3, 0.7), "2": [0.7, 1] |
| 56    | availability_predicted    | predicted availability by xgboost classification model                                                                              |
| 59-60 | *_revenue                 | lower and upper revenue range based on predicted price and using boundaries from availability predicted column                      |