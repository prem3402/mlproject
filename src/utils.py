from src.exception import CustomException
from src.logger import logging
import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_pickle(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param_grid = params.get(model_name)
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report

    except Exception as e:
        CustomException(e, sys)


def load_pickle(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)
