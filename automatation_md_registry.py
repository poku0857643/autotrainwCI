from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow_utils import preprocess_new_data, get_production_accuracy, train_and_evaluate, promote_model



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