import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
load_dotenv()


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup MLflow tracking and load model + test data"""
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "arkobera"
        repo_name = "PS5E10"  # ✅ current repository
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name)
        if not cls.model_version:
            raise ValueError(f"No registered versions found for model '{cls.model_name}'")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"
        print(f"Loading model from: {cls.model_uri}")
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        print("✅ Model loaded successfully")

        # Load validation dataset
        cls.valid_data = pd.read_csv("Artifacts/processed/valid_processed.csv")
        print(f"✅ Validation data loaded: {cls.valid_data.shape}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Production"):
        """Get latest version of a model from MLflow Model Registry"""
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        if not latest_versions:
            latest_versions = client.get_latest_versions(model_name)  # fallback
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        """Check that model loads without errors"""
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        """Check model input and output structure"""
        data = {
            "road_type": 1,
            "num_lanes": 1,
            "curvature": 0.78,
            "speed_limit": 45,
            "lighting": 2,
            "weather": 2,
            "road_signs_present": True,
            "public_road": False,
            "time_of_day": 2,
            "holiday": True,
            "school_season": False,
            "num_reported_accidents": 0,
            "y": 0.534,  # Not used in prediction, just to match structure
        }
        input_df = pd.DataFrame([data])

        # Predict using the loaded model
        prediction = self.model.predict(input_df)

        # Expected number of input features
        self.assertEqual(input_df.shape[1], 13)
        # Output should be numeric
        self.assertTrue(np.issubdtype(type(prediction[0]), np.number))

    def test_model_performance(self):
        """Evaluate model on holdout validation data"""
        X_valid = self.valid_data.drop(columns=['accident_risk'])
        y_valid = self.valid_data['accident_risk']

        # Predict using the new model
        y_pred = self.model.predict(X_valid)

        # Calculate regression metrics
        rmse = root_mean_squared_error(y_valid, y_pred)
        mae = mean_absolute_error(y_valid, y_pred)
        r2 = r2_score(y_valid, y_pred)

        print(f"\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Define minimal acceptable performance thresholds
        expected_rmse = 0.25
        expected_r2 = 0.2

        # Assert model performance
        self.assertLessEqual(rmse, expected_rmse, f"RMSE should be ≤ {expected_rmse}")
        self.assertGreaterEqual(r2, expected_r2, f"R² should be ≥ {expected_r2}")


if __name__ == "__main__":
    unittest.main()
