import schedule
import time
import os
from train import train_and_evaluate

def check_new_data_and_train():
    if os.path.exists("new_data.csv"):  # Check for new data
        print("New data found! Retraining...")
        train_and_evaluate()
        os.remove("new_data.csv")  # Clean up after processing

schedule.every(1).hour.do(check_new_data_and_train)  # Check every hour

while True:
    schedule.run_pending()
    time.sleep(1)
