# data preprocessing

import numpy as np
import pandas as pd
import scipy
import os
import warnings
from src.logger import logging
from src.exception import MyException
import sys
warnings.filterwarnings("ignore")


class Preprocessing:
    def __init__(self, train, test, org, valid):
        self.train = train.set_index("id")
        self.test = test.set_index("id")
        self.org = org
        self.target = "Personality"
        self.valid = valid.set_index("id")
    def impute_categorical(self):
        for col in self.test.select_dtypes(include=['object']).columns:
            self.train[col].fillna(self.train[col].mode()[0], inplace=True)
            self.test[col].fillna(self.test[col].mode()[0], inplace=True)
            self.org[col].fillna(self.org[col].mode()[0], inplace=True)
            self.valid[col].fillna(self.valid[col].mode()[0], inplace=True)
    def impute_numerical(self):
        for col in self.test.select_dtypes(include=['float64', 'int64']).columns:
            self.train[col].fillna(self.train[col].mean(), inplace=True)
            self.test[col].fillna(self.test[col].mean(), inplace=True)
            self.org[col].fillna(self.org[col].mean(), inplace=True)
            self.valid[col].fillna(self.valid[col].mean(), inplace=True)
    def encode_target(self):
        self.train[self.target] = self.train[self.target].map({"Extrovert":0,"Introvert":1})
        self.org[self.target] = self.org[self.target].map({"Extrovert":0,"Introvert":1})
        self.valid[self.target] = self.valid[self.target].map({"Extrovert":0,"Introvert":1})
    def encode_categorical(self):
        mapper = {"Yes": 1, "No": 0}
        for col in self.test.select_dtypes(include=['object']).columns:
            self.train[col] = self.train[col].map(mapper)
            self.test[col] = self.test[col].map(mapper)
            self.org[col] = self.org[col].map(mapper)
            self.valid[col] = self.valid[col].map(mapper)
    def preprocess(self):
        self.impute_categorical()
        self.impute_numerical()
        self.encode_categorical()
        self.encode_target()
        print("Preprocessing complete.")
        return self.train, self.test, self.org, self.valid
    

def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(f):
    def clip_f(X):
        sigma = 0.05
        mu = f(X)
        a, b = -mu/sigma, (1-mu)/sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu*(Phi_b-Phi_a)+sigma*(phi_a-phi_b)+1-Phi_b
    return clip_f


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./Artifacts/raw/train.csv')
        valid_data = pd.read_csv('./Artifacts/raw/valid.csv')
        logging.info('data loaded properly')

        # Transform the data
        z = clip(f)(train_data)
        train_data["y"] = z.values
        z = clip(f)(valid_data)
        valid_data["y"] = z.values
        # train_processed_data = preprocess_dataframe(train_data, 'review')
        # test_processed_data = preprocess_dataframe(test_data, 'review')
        CATS = [col for col in train_data.columns if train_data[col].dtype in ['O']]
        for col in CATS:
            train_data[col],_ = train_data[col].factorize()
            valid_data[col],_ = valid_data[col].factorize()

        # Store the data inside data/processed
        data_path = os.path.join("./Artifacts", "processed")
        os.makedirs(data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        # test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        # org_processed_data.to_csv(os.path.join(data_path, "original_processed.csv"), index=False)
        valid_data.to_csv(os.path.join(data_path, "valid_processed.csv"), index=False)

        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        raise MyException(e,sys) # type: ignore

if __name__ == '__main__':
    main()