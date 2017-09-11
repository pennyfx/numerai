#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""
import numpy as np
import pandas as pd
import math
import cPickle
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from xgboost import XGBClassifier
import random
random.seed(0)
# Force matplotlib to not use any Xwindows backend.

def main():
    # Set seed for reproducibility
    np.random.seed(0)

    training_data_path = os.path.join('data', 'numerai_training_data.csv')
    prediction_data_path = os.path.join('data', 'numerai_tournament_data.csv')
    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(training_data_path, header=0)
    prediction_data = pd.read_csv(prediction_data_path, header=0)


    # Some clean up. Replace #DIV/0! with 0
    # I think 0 is a reasonable, non-biasing number because if, e.g. #Months is 0, a spend per month of 0 is reasonable
    training_data.replace(to_replace='#DIV/0!',value='0',inplace=True)
    training_data.fillna(0, inplace=True)
    prediction_data.replace(to_replace='#DIV/0!',value='0',inplace=True)
    prediction_data.fillna(0, inplace=True)
    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]
    X_test = prediction_data[features]
    Y_test = prediction_data["target"]
    ids = prediction_data["id"]
    X.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1, inplace=True)

    le = LabelEncoder()
    training_data.reset_index( drop = True, inplace = True )
    Y = le.fit_transform(Y)
    X.fillna(0, inplace=True)
    
    #*** Split into training and testing data
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
    X_train.sort_index(axis=1, inplace=True)
    X_test.sort_index(axis=1,inplace=True)
    X_val.sort_index(axis=1,inplace=True)

    print("Training...")
    model = XGBClassifier() 
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)
    #*** Test
    test_acc = accuracy_score(y_val, y_val_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict_proba(X_test)
    results = y_test_pred[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(ids).join(results_df)

    # save the classifier
    stats = {"train accuracy": train_acc,"test accuracy":test_acc, 'label':'initial model',}
    # Save the predictions out to a CSV file
    predictions_path = os.path.join(os.environ['DATMO_TASK_DIR'], 'predictions.csv')
    joined.to_csv(predictions_path, index=False)
    model_filename = os.path.join(os.environ['DATMO_TASK_DIR'],'model.dat')
    pickle.dump(model, open(model_filename, 'wb'))
    stats_filename = os.path.join(os.environ['DATMO_TASK_DIR'],'stats.json')
    with open(stats_filename, 'wb') as f:
        f.write(json.dumps(stats))


if __name__ == '__main__':
    main()
