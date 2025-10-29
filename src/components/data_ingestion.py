import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

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
        logging.info("Data Ingestion process started")
        try:
            logging.info(f"Reading CSV file from {raw_data}")
            df = pd.read_csv(raw_data)
            logging.info(f"Dataset loaded with shape: {df.shape}")
            
            logging.info(f"Creating directory: {os.path.dirname(self.data_paths.train_data_path)}")
            os.makedirs(os.path.dirname(self.data_paths.train_data_path), exist_ok=True)
            
            logging.info(f"Saving raw data to {self.data_paths.raw_data_path}")
            df.to_csv(self.data_paths.raw_data_path, header=True)
            logging.info("Raw data successfully stored")

            logging.info("Initiating train-test split with test_size=0.2")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info(f"Split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")
            
            logging.info(f"Saving train data to {self.data_paths.train_data_path}")
            train_set.to_csv(self.data_paths.train_data_path, header=True)
            
            logging.info(f"Saving test data to {self.data_paths.test_data_path}")
            test_set.to_csv(self.data_paths.test_data_path, header=True)

            logging.info("Train-Test split data successfully saved to disk")

            return (
                self.data_paths.train_data_path,
                self.data_paths.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

