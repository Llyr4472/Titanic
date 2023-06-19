# %%
import numpy as np
import os
import sys
import tensorflow as tf


# %%
def main():

    #load data
    test_f = "test.csv"
    train_f = "train.csv"
    X_train, Y_train = load_data(train_f)
    X_test = load_data(test_f)

    #Train model and predict results
    model = train_model(X_train, Y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(Y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

# %%
def load_data(file):
    #TODO

# %%
def train_model(x,y):
    #TODO