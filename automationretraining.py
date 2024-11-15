import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def preprocess_new_data(file_path):
    # Replace with actual preprocessing logic
    import pandas as pd
    return pd.read_csv(file_path)

def train_and_evaluate():
    # Load data
    data = load_iris()  # Replace with your data source
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Start MLflow experiment
    mlflow.set_experiment("Iris Retraining Experiment")

    with mlflow.start_run():
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
        if acc > get_production_accuracy():
            mlflow.register_model("runs:/{}/model".format(mlflow.active_run().info.run_id), "IrisModel")
            print("New model registered!")

def get_production_accuracy():
    # Fetch production accuracy from MLflow Model Registry
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("IrisModel", stages=["Production"])[0]
    return latest_version.metrics["accuracy"]

if __name__ == "__main__":
    preprocess_new_data("new_data.csv")
    train_and_evaluate()
