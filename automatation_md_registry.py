from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

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

if __name__ == "__main__":
    client = MlflowClient()
    model_name = "IrisModel"
    production_accuracy = get_production_accuracy(model_name)

    # Check latest model version
    latest_version_info = client.search_model_versions(f"name='{model_name}'")[-1]
    latest_version = latest_version_info.version
    latest_run_id = latest_version_info.run_id
    latest_accuracy = client.get_run(latest_run_id).data.metrics["accuracy"]

    if latest_accuracy > production_accuracy:
        promote_model(model_name, latest_version)
    else:
        print("New model does not outperform the current production model.")