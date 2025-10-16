# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection
from src.exception import MyException
import sys
from dotenv import load_dotenv
# Load environment variables
load_dotenv()


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def save_data(train_data: pd.DataFrame, valid_data: pd.DataFrame,data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        # test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        valid_data.to_csv(os.path.join(raw_data_path, "valid.csv"), index=False)
        # original_data.to_csv(os.path.join(raw_data_path, "original.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        raise MyException(e,sys) # type: ignore

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        s3 = s3_connection.s3_operations(
            os.getenv("BUCKET_NAME"),
            os.getenv("AWS_ACCESS_KEY"),
            os.getenv("AWS_SECRET_KEY")
        )
        train = s3.fetch_file_from_s3("ps5e10_data.csv")
        # test = s3.fetch_file_from_s3("test.csv")
        # original_data = s3.fetch_file_from_s3("personality.csv")
        logging.info('Data loaded successfully from S3.')

        train, valid = train_test_split(train, test_size=test_size, random_state=42)
        save_data(train, valid,data_path='./Artifacts') #type: ignore
        logging.info('Data ingestion process completed successfully.')
    except Exception as e:
        raise MyException(e,sys) # type: ignore

if __name__ == '__main__':
    main()