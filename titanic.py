# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

# %%
EPOCHS = 20
TEST_SIZE = 0.4
DROPOUT = 0.5

# %%
def main():

    #load data
    train_f = "train.csv"
    evidence, label = load_data(train_f)
    x_train, x_test, y_train, y_test =  train_test_split(evidence,label,test_size=TEST_SIZE)

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

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
def get_model():
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

    #compile and return
    model.compile(
        optimizer="adam",
        loss = "binary_crossentropy",
        metrics = [tf.keras.metrics.Precision(),"accuracy"],
    )   
    return model

# %%
if __name__ == "__main__":
    main()