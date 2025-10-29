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
            logging.info("Starting the training pipeline")
            
            logging.info("Initializing Data Ingestion")
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

            logging.info("Initializing Data Transformation")
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.transformation(train_path=train_path, test_path=test_path)
            logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

            logging.info("Initializing Model Training")
            model = ModelTrainer()
            success = model.initiate_model_training(train_array=train_arr, test_array=test_arr)
            logging.info(f"Model training completed with R2 score: {success}")

            if success is not None:
                logging.info("Training pipeline completed successfully")
                return "yes"
            else:
                logging.warning("Training pipeline completed but no successful model was found")
                return "no"

        except Exception as e:

            raise CustomException(e, sys)


