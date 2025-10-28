import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

raw_data = "notebook/data/stud.csv"

@dataclass
class DataIngestionPath:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        self.data_paths = DataIngestionPath()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started...")
        try:
            logging.info("Reading CSV file...")
            df = pd.read_csv(raw_data)
            os.makedirs(os.path.dirname(self.data_paths.train_data_path), exist_ok=True)
            df.to_csv(self.data_paths.raw_data_path, header=True)
            logging.info("Raw data stored")

            logging.info("Train-Test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            train_set.to_csv(self.data_paths.train_data_path, header=True)
            test_set.to_csv(self.data_paths.test_data_path, header=True)

            logging.info("Train-Test split completed and stored")
            return (
                self.data_paths.train_data_path,
                self.data_paths.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
