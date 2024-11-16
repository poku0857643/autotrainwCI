# mlflow_utils.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

def preprocess_new_data(file_path):
    data = pd.read_csv(file_path)
    X_new = data.iloc[:, :-1]
    y_new = data.iloc[:, -1]
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
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        production_accuracy = get_production_accuracy()
        if acc > production_accuracy:
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "IrisModel")
            print("New model registered and promoted to production!")
        else:
            print("New model did not outperform production model.")
        return acc

def get_production_accuracy():
    """
    Fetch the accuracy of the current production model from the MLflow registry.
    Returns:
        float: Accuracy of the production model, or 0 if no production model exists.
    """
    client = MlflowClient()
    model_name = "IrisModel"

    try:
        # Fetch all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")

        # Find the production model version
        for version in model_versions:
            if version.current_stage == "Production":
                production_run_id = version.run_id

                # Fetch metrics for the production model
                run_data = client.get_run(production_run_id).data
                return run_data.metrics.get("accuracy", 0)

        print("No production model found.")
        return 0
    except Exception as e:
        print(f"Error fetching production model accuracy: {e}")
        return 0


def promote_model(model_name, version):
    """
    Promote the specified model version to the Production stage in the MLflow registry.
    Args:
        model_name (str): Name of the registered model.
        version (int): Version of the model to promote.
    """
    client = MlflowClient()

    try:
        # Transition the specified model version to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True  # Archive previous production versions
        )
        print(f"Model {model_name} version {version} promoted to Production.")
    except RestException as e:
        if "Registered Model with name" in str(e):
            print(f"Model {model_name} not found. Registering it as a new model.")
            client.create_registered_model(model_name)
        else:
            print(f"Error promoting model to Production: {e}")