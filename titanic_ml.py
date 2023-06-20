# %%
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# %%
NEIGHBOURS = 6 #best 80-84% at 6
EPOCHS = 10
TEST_SIZE = 0.25
DROPOUT = 0.30

# %%
def main():

    #load data
    train_f = "data/train.csv"
    evidence, label = load_data(train_f)
    x_train, x_test, y_train, y_test =  train_test_split(evidence,label,test_size=TEST_SIZE)

    #Train and evaluate KNeighbour classifier
    kn_predictions = train_kn_model(x_train,y_train,x_test)
    kn_accuracy = evaluate_accuracy(y_test, kn_predictions)
    print("kNeighbors Accuracy:",kn_accuracy)

    # Train and evaluate Random Forest classifier
    rf_predictions = train_random_forest(x_train, y_train, x_test)
    rf_accuracy = evaluate_accuracy(y_test, rf_predictions)
    print("Random Forest Accuracy:", rf_accuracy)

    # Train and evaluate Logistic Regression
    lr_predictions = train_logistic_regression(x_train, y_train, x_test)
    lr_accuracy = evaluate_accuracy(y_test, lr_predictions)
    print("Logistic Regression Accuracy:", lr_accuracy)

    # Train and evaluate Support Vector Machine (SVM)
    svm_predictions = train_support_vector_machine(x_train, y_train, x_test)
    svm_accuracy = evaluate_accuracy(y_test, svm_predictions)
    print("Support Vector Machine Accuracy:", svm_accuracy)

    #Train and evaluate using neural network
    nn_predictions = train_neural(x_train, y_train, x_test, y_test)

    #combine all the predictions
    predictions = combine_predictions(kn_predictions,rf_predictions,lr_predictions,svm_predictions, nn_predictions)
    sensitivity, specificity = evaluate(y_test,predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"Combined Accuracy: {50* (sensitivity+specificity):.2f}%")

# %%
def load_data(filename):

    #load data
    data = pd.read_csv(filename)

    #check for missing data
    data.isnull().sum()

    # Fill missing values in the "Age" column with the median
    data["Age"].fillna(0, inplace=True)

    # One-hot encode the columns
    data = pd.get_dummies(data, columns=["Sex", "Embarked"])

    # Convert boolean columns to integers
    boolean_columns = ["Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    data[boolean_columns] = data[boolean_columns].astype(int)


    # Perform min-max scaling on the "Age" and "Fare" columns
    data["Age"] = (data["Age"] - data["Age"].min()) / (data["Age"].max() - data["Age"].min())
    data["Fare"] = (data["Fare"] - data["Fare"].min()) / (data["Fare"].max() - data["Fare"].min())

    # Select features and target
    features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    target = "Survived"

    data.to_csv(filename.split(".")[0]+"panda.csv", index=True)

    x = data[features]
    y = data[target]
    return x, y



# %%
def evaluate(labels, predictions):
    sensitivity, specificity = 0.0, 0.0
    p_labels = labels.value_counts().get(1)
    n_labels = len(labels) -  p_labels
    for prediction,label in zip(predictions,labels):
        if label == 1 and prediction == label:
            sensitivity += 1/p_labels
        elif label == 0 and prediction == label:
            specificity += 1/n_labels
    return sensitivity,specificity

def evaluate_accuracy(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# %%
def train_random_forest(x_train, y_train, x_test):
    # Train Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    rf_predictions = rf_classifier.predict(x_test)
    return rf_predictions

def train_logistic_regression(x_train, y_train, x_test):
    # Train Logistic Regression
    lr_classifier = LogisticRegression(random_state=42)
    lr_classifier.fit(x_train, y_train)

    # Make predictions on the test set
    lr_predictions = lr_classifier.predict(x_test)
    return lr_predictions

def train_support_vector_machine(x_train, y_train, x_test):
    # Train Support Vector Machine (SVM)
    svm_classifier = SVC(random_state=42)
    svm_classifier.fit(x_train, y_train)

    #save model
    with open("model_svm.h5", "w") as file:
        pickle.dump(svm_classifier, file)

    # Make predictions on the test set
    svm_predictions = svm_classifier.predict(x_test)
    return svm_predictions


def train_kn_model(evidence, labels,x_test):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    #train and predict
    model=KNeighborsClassifier(n_neighbors=NEIGHBOURS)
    model.fit(evidence,labels)
    predictions = model.predict(x_test)

    return predictions


def train_neural(x_train, y_train, x_test, y_test):
    #Train network
    model =tf.keras.Sequential([

        #hidden layer
        tf.keras.layers.Dense(256,input_shape=(10,), activation="relu"),
        tf.keras.layers.Dense(128,input_shape=(10,), activation="relu"),
        tf.keras.layers.Dense(64,input_shape=(10,), activation="relu"),

        #dropout layer
        tf.keras.layers.Dropout(DROPOUT),

        #output layer
        tf.keras.layers.Dense(1,activation="sigmoid"),
    ])

    #compile and fit network
    model.compile(
        optimizer="adam",
        loss = "binary_crossentropy",
        metrics = [tf.keras.metrics.Precision(),"accuracy"],
    )
    
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    model.evaluate(x_test,  y_test, verbose=1)

    #save model
    model.save("model_nn.h5")

    #return predictions
    return model.predict(x_test)
    
# %%
def combine_predictions(dat1,dat2,dat3,dat4,dat5):
    predictions = []
    for p1,p2,p3,p4,p5 in zip(dat1,dat2,dat3,dat4,dat5):
        if p1 + p2 + p3 + p4 + p5 > 2 :
            predictions.append(1)
        else :
            predictions.append(0)
    return predictions

# %%
if __name__ == "__main__":
    main()