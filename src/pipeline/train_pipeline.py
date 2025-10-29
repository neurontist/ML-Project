import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class Training:
    def __init__(self):
        pass

    def fit(self):
        try: 
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.transformation(train_path=train_path, test_path=test_path)

            model = ModelTrainer()
            success = model.initiate_model_training(train_array=train_arr, test_array=test_arr)

            if success is not None:
                return "yes"
            else:
                return "no"

        except Exception as e:

            raise CustomException(e, sys)


