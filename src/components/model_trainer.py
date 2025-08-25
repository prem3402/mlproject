import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_pickle, evaluate_model

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_obj_file = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_dataset, test_dataset):
        try:
            logging.info("Split train and test input data")
            X_train, y_train, X_test, y_test = (
                train_dataset[:, :-1],
                train_dataset[:, -1],
                test_dataset[:, :-1],
                test_dataset[:, -1],
            )
            models = {
                "linear regression": LinearRegression(),
                "Decisiontree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "Adaboost": AdaBoostRegressor(),
                "Gradiantboost": GradientBoostingRegressor(),
                "XGboost": XGBRegressor(),
                "Catboost": CatBoostRegressor(verbose=False),
            }
            params = {
                "Decisiontree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradiantboost": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "linear regression": {},
                "XGboost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Catboost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "Adaboost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }
            model_reports: dict = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            best_model_name, best_model_score = max(
                model_reports.items(), key=lambda x: x[1]
            )
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("best model found on both train and test dataset")
            save_pickle(
                file_path=self.model_trainer_config.trained_model_obj_file,
                obj=best_model,
            )
            y_pred = best_model.predict(X_test)
            score = r2_score(y_test, y_pred)
            return score

        except Exception as e:
            CustomException(e, sys)
