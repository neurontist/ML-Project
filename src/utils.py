import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params, cv=3, n_jobs=3, verbose=False):
    try:
        report = {}
        logging.info("Into model eval")

        # iterate by name so we can pick the matching param grid per model
        for model_name, model in models.items():
            # get only the param grid for this model (default to empty dict/list)
            param_grid = params.get(model_name, {})
            if not isinstance(param_grid, (dict, list)):
                param_grid = {}

            gs = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring="r2")
            gs.fit(X_train, y_train)

            best_estimator = gs.best_estimator_
            best_estimator.fit(X_train, y_train)

            train_pred = best_estimator.predict(X_train)
            train_score = r2_score(y_true=y_train, y_pred=train_pred)

            test_pred = best_estimator.predict(X_test)
            test_score = r2_score(y_true=y_test, y_pred=test_pred)

            report[model_name] = test_score

        logging.info("Returning report")
        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

