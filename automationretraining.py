
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from mlflow_utils import preprocess_new_data, get_production_accuracy, train_and_evaluate
from sklearn.model_selection import GridSearchCV
import mlflow
import mlflow.sklearn

# Define accuracy threshold
ACCURACY_THRESHOLD = 1.0

# Grid search parameter grid
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}



if __name__ == "__main__":
    # Load initial data
    print("Running automated retraining...")
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Check if new data is available
    new_data_path = "new_data.csv"
    if os.path.exists(new_data_path):
        print("New data found, training with new data.")
        X_new, y_new = preprocess_new_data(new_data_path)
        X_train = pd.concat([pd.DataFrame(X_train), X_new])
        y_train = pd.concat([pd.Series(y_train), y_new])
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, use_grid_search=True)
    else:
        print("No new data found, checking production accuracy.")
        # Train with current data if production accuracy doesn't meet threshold
        production_accuracy = get_production_accuracy()
        if production_accuracy < ACCURACY_THRESHOLD:
            print(f"Production accuracy {production_accuracy} is below threshold {ACCURACY_THRESHOLD}. Retraining with current data.")
            accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, use_grid_search=True)
        else:
            print("Production model meets accuracy threshold. No retraining needed.")

