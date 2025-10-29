import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            features_transformed = preprocessor.transform(features)
            
            preds = model.predict(features_transformed)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def format_data(self):

        feature_dict = {
             "gender": [self.gender],
            "race_ethnicity": [self.race_ethnicity],
            "parental_level_of_education": [self.parental_level_of_education],
            "lunch": [self.lunch],
            "test_preparation_course": [self.test_preparation_course],
            "reading_score": [self.reading_score],
            "writing_score": [self.writing_score],
        }
        try:
            df = pd.DataFrame(feature_dict)

            return df
        except Exception as e:
            raise CustomData(e, sys)



