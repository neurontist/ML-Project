import os
import sys

import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score

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
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    report = {}

    logging.info("Into model eval")

    for i in range(len(list(models))):
        model = list(models.values())[i]

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_score = r2_score(y_true=y_train, y_pred=train_pred)

        test_pred = model.predict(X_test)
        test_score = r2_score(y_true=y_test, y_pred=test_pred)

        report[list(models.keys())[i]] = test_score

    logging.info("Returing report")

    return report




