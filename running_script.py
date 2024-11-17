import subprocess
import time

def start_mlflow():
    """Start the MLflow server."""
    return subprocess.Popen([
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5001"
    ])

def start_ngrok():
    """Start ngrok to expose the MLflow server."""
    return subprocess.Popen(["ngrok", "http", "5001"])

if __name__ == "__main__":
    print("Starting MLflow server...")
    mlflow_process = start_mlflow()
    time.sleep(5)  # Give MLflow server time to start

    print("Starting ngrok...")
    ngrok_process = start_ngrok()

    print("MLflow server and ngrok are running. Press Ctrl+C to stop.")

    try:
        mlflow_process.wait()
        ngrok_process.wait()
    except KeyboardInterrupt:
        print("Stopping services...")
        mlflow_process.terminate()
        ngrok_process.terminate()
