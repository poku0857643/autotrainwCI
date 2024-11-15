from mlflow.tracking import MlflowClient

def promote_model(model_name, version):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Model {model_name} version {version} promoted to Production.")
