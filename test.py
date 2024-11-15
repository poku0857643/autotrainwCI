from train import train_and_evaluate

def test_model_accuracy():
    acc = train_and_evaluate()
    assert acc > 0.8  # Ensure accuracy is above threshold
