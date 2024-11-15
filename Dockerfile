FROM python:3.9-slim

RUN pip install mlflow scikit-learn

CMD ["mlflow", "models", "serve", "-m", "models:/IrisModel/Production", "--port", "5001"]
