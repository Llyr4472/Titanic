import sklearn
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

MODEL = "model.h5"
test_x = "data/test.csv"

def main():
    model = tf.keras.models.load_model('model.h5')

    evidence = load_data(test_x)

    predictions = model.predict(evidence).flatten()
    predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]

    passenger_ids = pd.read_csv(test_x)['PassengerId']

    result_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    result_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to file")
        


def load_data(filename):

    #load data
    data = pd.read_csv(filename)

    #Check for empty data
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

    x = data[features]
    return x

if __name__ == "__main__":
    main()