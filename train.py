import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from mlflow_utils import preprocess_new_data, get_production_accuracy, train_and_evaluate, promote_model



# MLflow Tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5001")


# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Iris Experiment")



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

