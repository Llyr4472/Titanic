import pandas as pd

def check_predictions(predictions_file, ground_truth_file):
    predictions = pd.read_csv(predictions_file)['Survived']
    ground_truth = pd.read_csv(ground_truth_file)['Survived']

    accuracy = (predictions == ground_truth).mean()
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    predictions_file = "data/predictions.csv"
    ground_truth_file = "data/gender_submission.csv"
    check_predictions(predictions_file, ground_truth_file)