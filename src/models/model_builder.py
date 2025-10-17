import numpy as np
import pandas as pd
import pickle
from lightgbm import LGBMRegressor # type: ignore
import os
import yaml
from src.logger import logging
from src.exception import MyException
import sys


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
        
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def train_model(X_train: np.ndarray, y_train: np.ndarray, model: LGBMRegressor):
    """Train the Logistic Regression model."""
    try:
        model.fit(X_train, y_train)
        logging.info('Model training completed')
        return model
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def main():
    try:

        train_data = load_data('./Artifacts/processed/train_processed.csv').head(10000)
        X_train = train_data.copy()
        y_train = X_train.pop("accident_risk")

        params_lgbm = {
            'verbose': -1,
            'random_state': 1,
            'objective': 'regression',
            'n_estimators': 4100,
            'learning_rate': 0.01,
            'colsample_bytree': 0.6,
            'max_depth': 8,
            'max_bin': 5000,
                    }

        
        model = LGBMRegressor(**params_lgbm)
        reg = train_model(X_train, y_train, model) #type: ignore
        
        save_model(reg, 'model/model.pkl')
    except Exception as e:
        raise MyException(e,sys) # type: ignore

if __name__ == '__main__':
    main()