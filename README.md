# How Price and Availability Impact Airbnb Revenue
Developer:  Aarushi Verma, Cindy Chiu, Dauren Bizhanov, Sydney Donati-Leach

## Project Objectives 
This repository is used to tell a host or an investor what factors to increase the success of an Airbnb listing. 

## Requirements 
This project is built with Python 3, and the visualizations are created using Jupyter Notebooks. In order to run all the files without error, please pip install all the packages in the requirements.txt. 

## Data: 
The data can be retrieved from http://insideairbnb.com/get-the-data website. The city we’re training in is Hawaii, USA. Since the data is huge and zipped, the user can clone the github repository and run the download_data.py under the source folder. Running the `download_data.py` on the local machine will retrieve both training and testing data from the website and create a copy on your local machine. The files will be saved under the same folder as the script with 3 separate folders: `Train`, `Test_Broward` and `Test_Create`. Users can open the files on their local machine. 

## Pre-processing
After running the `download_data.py` file, users can perform data pre-processing on both training data and testing data using the `preprocessing.py` file under the source folder. Running the `preprocessing.py` will perform feature transformations on both the training set and testing files. 

## Training the Models
In order to replicate the results, please use the `train_models_classification.py` and `train_models_regression.py` under the source folder. Those two files contain multiple models that we have implemented for predictions. The `train_models_regression.py` includes RandomForest Regressor, XGBoost Regressor and LightBGM Regressor. The `train_models_classification.py` includes RandomForest Classifier, XGBoost Classifier and LightBGM Classifier Running these file will also save the model results as JSON files to Models folder. In addition, the evaluation metrics of the validation data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the Data folder

## Model Prediction
In order to evaluate the performance on the test cities (Broward County and Crete), use can run the `predictions_on_test_data.py` under the source folder. We leverage the model that performed the best on the validation set and predict it on the testing data. The evaluation metrics of the testing data, including precision, recall, confusion matrix, RMSE and MAE, are saved in the Data folder. 

