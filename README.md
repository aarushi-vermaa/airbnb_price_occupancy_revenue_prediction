# Investing with Airbnb: Predicting Price and Availability of Listings

Developers:  Aarushi Verma, Cindy Chiu, Dauren Bizhanov, Sydney Donati-Leach

## Project Objectives
This project aims to help potential property investors to estimate their annual revenue. Our model aims to predict the average daily price and occupancy rate bracket for the next year for a listing using Airbnb data. By evaluating feature importance we also aim to help prospective hosts to increase the success of an Airbnb listing. 

## Data
The data can be retrieved from the http://insideairbnb.com/get-the-data website. The city we’re training our model on is Hawaii, USA. Since the data is huge and zipped, the user can clone the github repository and run the download_data.py under the source folder. Running the `download_data.py` on the local machine will retrieve both training and testing data from the website and create a copy on your local machine. The files will be saved under the same folder as the script with 3 separate folders: `Train`, `Test_Broward` and `Test_Create`. Users can open the files on their local machine.

## Conclusion
Home-sharing services like Airbnb can become a viable source of passive income if the host chooses the correct price point and the listing is booked regularly. Previous studies focused on predicting listing prices and few focused on availability of a listing. Availability can be an important factor in predicting revenue as the listing will be profitless without any bookings. This research builds a model that is able to predict the price of a listing as well as anticipated availability based on Airbnb data. Investors could do a simple calculation of the annual occupancy rate multiplied by the nightly price to obtain the annual revenue of a listing. In this study, we used XGBoost to train on Hawaii data. We also perform cross-domain testing in Broward County, Florida and Crete, Greece. The price model performed well in both US locations while the performance of the availability model was limited.y


## Requirements 
This project is built with Python 3, and the visualizations are created using Jupyter Notebooks. In order to run all the files without error, please follow the following steps:

Step 1: Git clone the repo on your local machine    
```
git clone https://github.com/aarushi-vermaa/705_FinalProject.git
```

Step 2: Install required packages   
``` 
pip install -r requirements.txt
```

Step 3: Download the data by running the `download_data.py` file by navigating to the `/00_setup` folder  

``` 
python3 /00_setup/download_data.py
```

Running this script retrieve both training and testing data from the website and create a copy on your local machine. The files will be saved under the same folder as the script with 3 separate folders: `Train`, `Test_Broward` and `Test_Create`. Users can access the files on their local machine. These source files can also be found in `/01_source_data/` folder. 

Step 4: Preprocess the train and test data    
```
python3 /00_setup/preprocessing.py
```

Running the `preprocessing.py` in `/00_setup` folder will perform feature transformations on both the training set and testing files.

Step 5: Train the models    

For the price prediction model:   
```
python3 /00_setup/train_models_regression.py
```

For the occupancy prediction:   
```
python3 /00_setup/train_models_classification.py
```

Those two files contain multiple models that we have implemented for predictions. The `train_models_regression.py` includes RandomForest Regressor, XGBoost Regressor and LightBGM Regressor. The `train_models_classification.py` includes RandomForest Classifier, XGBoost Classifier and LightBGM Classifier Running these files will also save the model results as JSON files to `/30_results/Models` folder. In addition, the evaluation metrics of the validation data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the `/30_results/Model Results` folder
 
Step 6: Test Model performance    
```
python3 /10_code/predictions_on_test_data.py
```

We leverage the model that performed the best on the validation set and predict it on the testing data (Broward County and Crete). The evaluation metrics of the testing data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the `/30_results/Model Results` folder.

Step 7: Visualize Model performance
```
python3 /10_code/model_assessment_plots.py
python3 /10_code/confusion matrix.py.py

```

We can further visualize those testing results and validation results using plots. This includes confusion matrix for occupancy prediction and residual plot for price prediction. These plots can be found in `30_results/Plots` folder. 

Step 8: Generate Revenue 
```
python3 /10_code/predictions_on_test_data.py
```
We leveraged the model that performed the best on validation set for both occupancy and price prediction. Then we multiply the results and generate a range of revenue per year for the user. The revenue results are saved in the `/20_intermediate_files` folder.  


## Project Outcome
The detailed research paper can be found [here](https://github.com/aarushi-vermaa/705_FinalProject/blob/main/40_report/Final%20Report.pdf)


The video of our project presentation can be found [here](https://youtu.be/DqS0XG79uHE)
