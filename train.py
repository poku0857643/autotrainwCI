import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Iris Experiment")


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

    # Ensure the active run ID is used
    if acc > get_production_accuracy():
        run_id = run.info.run_id  # Use the run ID from the active context
        model_uri = f"runs:/{run_id}/model"
        registered_model = mlflow.register_model(model_uri, "IrisModel")
        new_version = registered_model.version

        # Promote the new model
        promote_model("IrisModel", new_version)

