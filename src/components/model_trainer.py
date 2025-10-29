import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modelTrainerconfig = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Starting model training process")
            logging.info("Splitting arrays into features and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

            models = {
                    "Random Forest": RandomForestRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    # "Linear Regression": LinearRegression(),
                    "XGBRegressor": XGBRegressor(),
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                }     

            params = {
                "Decision Tree": [
                    {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']}
                ],
                "Random Forest": [
                    {'n_estimators': [8,16,32,64,128,256]}
                ],
                "Gradient Boosting": [
                    {
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'n_estimators': [8,16,32,64,128,256]
                    }
                ],
                # "Linear Regression": [
                #     {}
                # ],
                "XGBRegressor": [
                    {'learning_rate':[.1,.01,.05,.001],
                     'n_estimators': [8,16,32,64,128,256]}
                ],
                "CatBoosting Regressor": [
                    {'depth': [6,8,10],
                     'learning_rate': [0.01, 0.05, 0.1],
                     'iterations': [30, 50, 100]}
                ],
                "AdaBoost Regressor": [
                    {'learning_rate':[.1,.01,0.5,.001],
                     'n_estimators': [8,16,32,64,128,256]}
                ]
            }

            
            logging.info("Creating model score dictionary...")

            logging.info("Starting model evaluation process...")
            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            logging.info("Model evaluation completed. Results:")
            for model_name, score in model_report.items():
                logging.info(f"{model_name}: R2 Score = {score}")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            
            logging.info(f"Best performing model: {best_model_name} with R2 score: {best_model_score}")
            
            if best_model_score<0.6:
                logging.error(f"No model achieved minimum required R2 score of 0.6. Best score: {best_model_score}")
                raise CustomException("No best model found")
            
            best_model.fit(X_train, y_train)

            save_object(
                filepath=self.modelTrainerconfig.model_trainer_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
        