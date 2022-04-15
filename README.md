# Investing with Airbnb: Predicting Price and Availability of Listings

Developers:  Aarushi Verma, Cindy Chiu, Dauren Bizhanov, Sydney Donati-Leach

## Project Objectives
This project aims to help potential property investors to estimate their annual revenue. Our model aims to predict the average daily price and occupancy rate bracket for the next year for a listing using Airbnb data. By evaluating feature importance we also aim to help prospective hosts to increase the success of an Airbnb listing. 

## Data
The data can be retrieved from the http://insideairbnb.com/get-the-data website. The city we’re training our model on is Hawaii, USA. Since the data is huge and zipped, the user can clone the github repository and run the download_data.py under the source folder. Running the `download_data.py` on the local machine will retrieve both training and testing data from the website and create a copy on your local machine. The files will be saved under the same folder as the script with 3 separate folders: `Train`, `Test_Broward` and `Test_Create`. Users can open the files on their local machine.

## Conclusion
Our price model using XGBoost Regressor performed well on predicting Airbnb listing prices in Hawaii and Broward County, USA. It is not generalizable outside the US as it performed much worse in Crete, Greece. Important features for predicting listing prices were space dependent such as how many guests can be accommodated and how many bathrooms there are in the property. 
For predicting yearly availability, our model using XGBoost Classifier performed moderately well. There was a lot of variability within the medium occupancy class due to the lack of ability to identify listings close to the cutoff point. This also means the model was not able to perform well in Broward County or Crete.  Therefore, our model can be used to predict revenue with lower and upper bounds only in Hawaii. 
There are a few limitations in this analysis. First, the data only provided the insight for booking 365 days in advance, and did not include any historical information. Airbnb guests do not book their stays far in advance, so we would not be able to capture the true occupancy rate until the booking date has passed. Furthermore, there is no clear identifier in the data to distinguish whether a listing was booked or blocked by the host. Future work can incorporate methodologies from other studies such as the distance from the city center which can impact price. Also, access to data over multiple years could improve the price model to better capture the seasonality changes, and improve the availability model to achieve better accuracy


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
python3 download_data.py
```

Running this script retrieve both training and testing data from the website and create a copy on your local machine. The files will be saved under the same folder as the script with 3 separate folders: `Train`, `Test_Broward` and `Test_Create`. Users can access the files on their local machine. 

Step 4: Preprocess the train and test data    
```
python3 preprocessing.py
```

Running the `preprocessing.py` will perform feature transformations on both the training set and testing files.

Step 5: Train the models    

For the price prediction model:   
```
python3 train_models_regression.py
```

For the occupancy prediction:   
```
python3 train_models_classification.py
```

Those two files contain multiple models that we have implemented for predictions. The `train_models_regression.py` includes RandomForest Regressor, XGBoost Regressor and LightBGM Regressor. The `train_models_classification.py` includes RandomForest Classifier, XGBoost Classifier and LightBGM Classifier Running these files will also save the model results as JSON files to `/30_results/Models` folder. In addition, the evaluation metrics of the validation data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the `/30_results/Model Results` folder
 
Step 6: Test Model performance    
```
python3 predictions_on_test_data.py
```

We leverage the model that performed the best on the validation set and predict it on the testing data (Broward County and Crete). The evaluation metrics of the testing data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the `\Data` folder.  

## Project Outcome
The detailed research paper can be found in [here] (https://github.com/aarushi-vermaa/705_FinalProject/blob/main/40_report/Final%20Report.pdf)


The video of our project presentation can be found here: https://youtu.be/DqS0XG79uHE
