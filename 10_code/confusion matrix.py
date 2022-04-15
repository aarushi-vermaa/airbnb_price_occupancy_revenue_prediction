import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

def confusion_matrix_validation():
    """
    This function generates a confusion matrix with all the models 
    we have used in the availablity classification on the validation set. 
    """
    # read in the model performance file we generated
    clf_model = pd.read_csv('../30_results/Model Results/clf_val_conf_matrix.csv')
    fig , ax = plt.subplots(2,3, figsize=(16,12))
    grid = plt.GridSpec(2, 3, )
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    ax3 = plt.subplot(grid[0, 2])
    ax4 = plt.subplot(grid[1, 0])
    ax5 = plt.subplot(grid[1, 1])
    axes = [ax1, ax2, ax3, ax4, ax5]

    # for each model, we want to plot the confusion matrix separately
    for i in range(len(clf_model.model.unique())):
        idx = i*3
        model_cf = np.array(clf_model.iloc[idx:idx+3, 0:3])
        # this is to help generate row wise percentage 
        # for each predicted class, how many percentage did we get correctly 
        sns.heatmap( np.true_divide(model_cf, model_cf.sum(axis=1, keepdims=True)), annot=True, 
                fmt='.2%', cmap='Blues',ax =axes[i])
        axes[i].set_title(clf_model.model.unique()[i], fontsize = 12)
        axes[i].set_xlabel('Actual Class', fontsize=10)
        axes[i].set_ylabel('Predicted Class', fontsize=10)
        plt.gcf().tight_layout()
    # delete the extra grid 
    fig.delaxes(plt.subplot(grid[1, 2]))
    plt.suptitle('Confusion Matrix of Different Models with Validation Data', y = 1, fontsize = 14)
    plt.savefig('confusion matrix validation.png')

def confusion_matrix_test(clf_model:pd.DataFrame):
    """
    This function generates a confusion matrix with the XGBoost model
    we have used in the availablity classification on the testing data. 
    """
    # read in the model performance file we generated
    fig , axes = plt.subplots(1,2, figsize=(9,5))
    # loop through each test cities 
    for i in range(len(clf_model.city.unique())):
        idx = i*3
        model_cf = np.array(clf_model.iloc[idx:idx+3, 0:3])
        # this is to help generate row wise percentage 
        # for each predicted class, how many percentage did we get correctly 
        sns.heatmap( np.true_divide(model_cf, model_cf.sum(axis=1, keepdims=True)), annot=True, 
                fmt='.2%', cmap='Blues',ax =axes[i])
        axes[i].set_title(clf_model.city.unique()[i], fontsize = 12)
        axes[i].set_xlabel('Actual Class', fontsize=10)
        axes[i].set_ylabel('Predicted Class', fontsize=10)
        plt.gcf().tight_layout()
    plt.suptitle('Confusion Matrix of XGBoost Classifier with Testing Data', y = 1, fontsize = 14)
    plt.savefig('confusion matrix test.png')


clf_test = pd.read_csv('../30_results/Model Results/clf_test_conf_matrix.csv')
confusion_matrix_validation()
confusion_matrix_test(clf_test)
