import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    pkl_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.dataTransformationObj = DataTransformationConfig()

    def initiate_preprocess(self):

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                    "gender",
                    "race_ethnicity",
                    "parental_level_of_education",
                    "lunch",
                    "test_preparation_course",]

            logging.info("Creating Pipeline...")

            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())  
                ]
            )

            cat_pipeline=Pipeline(
                    steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                    ]
                )
            
            preprocessor = ColumnTransformer(
                [
                ("numericalTransformation",num_pipeline, numerical_columns),
                ("categoricalTransformation",cat_pipeline, categorical_columns)
                ]
                    
            )

            logging.info("Pipelines Created...")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def transformation(self,train_path, test_path):
        try:
            logging.info("Train-Test Data Loading...")

            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Preprocessor Loading...")

            preprocessor = self.initiate_preprocess()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            train_input_data = train_data.drop(columns=[target_column_name], axis=1)
            train_output_data = train_data[target_column_name]

            test_input_data = test_data.drop(columns=[target_column_name], axis=1)
            test_output_data = test_data[target_column_name]

            logging.info("Transforming train data")
            transformed_train_input_data = preprocessor.fit_transform(train_input_data)
            logging.info("Transforming test data")
            transformed_test_input_data = preprocessor.transform(test_input_data)

            train_arr = np.c_[
                transformed_train_input_data, np.array(train_output_data)
            ]

            test_arr = np.c_[
                transformed_test_input_data, np.array(test_output_data)
            ]

            logging.info("Saving Preprocessor Pickle File")
            save_object(
                filepath=self.dataTransformationObj.pkl_file_path,
                obj=preprocessor
            )

            logging.info("Returning the transformed data...")
            return (
                train_arr,
                test_arr,
                self.dataTransformationObj.pkl_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
