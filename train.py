import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlflow_utils import preprocess_new_data, get_production_accuracy, train_and_evaluate, promote_model
import os


# MLflow Tracking URI
mlflow.set_tracking_uri("https://08ee-36-230-63-22.ngrok-free.app")



# Explicitly set the experiment name
experiment_name = "Iris Experiment1"
mlflow.set_experiment(experiment_name)

#experiment = mlflow.get_experiment_by_name(experiment_name)
# if experiment is None:
#     # Create the experiment if it doesn't exist
#     experiment_id = mlflow.create_experiment(experiment_name)
#     print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
# else:
#     # Use the existing experiment ID
#     experiment_id = experiment.experiment_id
#     print(f"Using existing experiment '{experiment_name}' with ID: {experiment_id}")

# Create a writable artifact directory
artifact_location = os.path.expanduser("~/mlflow-artifacts")
os.makedirs(artifact_location, exist_ok=True)

mlflow.set_tracking_uri(f"file://{artifact_location}")

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLflow experiment
# mlflow.set_experiment("Iris Experiment")



with mlflow.start_run() as run:
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Log parameters and metrics
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Ensure the active run ID is used
    if acc > get_production_accuracy():
        run_id = run.info.run_id  # Use the run ID from the active context
        model_uri = f"runs:/{run_id}/model"
        print(f"Registering model with URI: {model_uri}")

        # Register the model to the MLflow Model Registry
        registered_model = mlflow.register_model(model_uri, "IrisModel")
        new_version = registered_model.version

        # Promote the new model
        promote_model("IrisModel", new_version)
        print(f"New model version {new_version} promoted to Production.")
    else:
        print(f"Model accuracy ({acc}) did not surpass production accuracy.")
