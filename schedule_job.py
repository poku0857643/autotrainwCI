import os
from train import train_and_evaluate
# schedule_job.py
import schedule
import time
import subprocess

def retrain():
    print("Running automated retraining...")
    subprocess.run(["python", "automationretraining.py"])

# Schedule the job every day at midnight (you can adjust as needed)
schedule.every().day.at("00:00").do(retrain)

while True:
    schedule.run_pending()
    time.sleep(60)


def check_new_data_and_train():
    if os.path.exists("new_data.csv"):  # Check for new data
        print("New data found! Retraining...")
        train_and_evaluate()
        os.remove("new_data.csv")  # Clean up after processing

schedule.every(1).hour.do(check_new_data_and_train)  # Check every hour

while True:
    schedule.run_pending()
    time.sleep(1)
