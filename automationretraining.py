import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Define accuracy threshold
ACCURACY_THRESHOLD = 0.9

def preprocess_new_data(file_path):
    # Replace this with actual preprocessing logic
    data = pd.read_csv(file_path)
    # Assuming the CSV has columns for features and target
    X_new = data.iloc[:, :-1]  # Features (all columns except last)
    y_new = data.iloc[:, -1]   # Target (last column)
    return X_new, y_new

def get_production_accuracy():
    client = mlflow.tracking.MlflowClient()
    try:
        latest_version = client.get_latest_versions("IrisModel", stages=["Production"])[0]
        production_run_id = latest_version.run_id
        run_data = client.get_run(production_run_id).data
        return run_data.metrics.get("accuracy", 0)
    except IndexError:
        print("No production model found.")
        return 0

def train_and_evaluate(X_train, y_train, X_test, y_test):
    with mlflow.start_run() as run:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log parameters and metrics
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Check and register model if it outperforms current production
        production_accuracy = get_production_accuracy()
        if acc > production_accuracy:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "IrisModel")
            print("New model registered and promoted to production!")
        else:
            print("New model did not outperform production model.")
        return acc

if __name__ == "__main__":
    # Load initial data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Check if new data is available
    new_data_path = "new_data.csv"
    if os.path.exists(new_data_path):
        print("New data found, training with new data.")
        X_new, y_new = preprocess_new_data(new_data_path)
        X_train = pd.concat([pd.DataFrame(X_train), X_new])
        y_train = pd.concat([pd.Series(y_train), y_new])
        accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
    else:
        print("No new data found, checking production accuracy.")
        # Train with current data if production accuracy doesn't meet threshold
        production_accuracy = get_production_accuracy()
        if production_accuracy < ACCURACY_THRESHOLD:
            print(f"Production accuracy {production_accuracy} is below threshold {ACCURACY_THRESHOLD}. Retraining with current data.")
            accuracy = train_and_evaluate(X_train, y_train, X_test, y_test)
        else:
            print("Production model meets accuracy threshold. No retraining needed.")
