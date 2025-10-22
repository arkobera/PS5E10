# PS5E10: Playground Competition MLOps

This repository contains the full **MLOps pipeline** for the PS5E10 Kaggle Playground competition. The project demonstrates a production-ready workflow for data ingestion, feature engineering, model training, evaluation, and deployment using **DVC, MLflow, Streamlit, and GitHub Actions**.
Link to Deployment: https://ps5e10-ktr9ed72wvfz7hh8lwdlwu.streamlit.app/
-----

## 📂 Project Structure
```
.
PS5E10/
├── src/
│   ├── data/                # Data ingestion and feature engineering scripts
│   ├── models/              # Model building, evaluation, and registry scripts
│   └── connections/         # Connection scripts (e.g., S3, DagsHub)
├── Artifacts/               # Raw and processed datasets (tracked via DVC)
├── model/                   # Trained model artifacts
├── reports/                 # Metrics and experiment information
├── tests/                   # Unit and integration tests
├── requirements.txt         # Python dependencies
├── dvc.yaml                 # DVC pipeline stages
├── .github/workflows/       # GitHub Actions CI/CD pipeline
└── README.md                # Project documentation
````

-----

## 🚀 Features

- **Data Ingestion:** Pulls raw data from S3 or local storage.
- **Feature Engineering:** Cleans and transforms raw datasets into model-ready features.
- **Model Training:** Trains ML models with reproducible pipelines using DVC.
- **Model Evaluation:** Computes metrics and stores results in `reports/`.
- **Model Registry:** Promotes the best model to production using MLflow and DagsHub.
- **CI/CD Pipeline:** GitHub Actions workflow runs DVC repro, tests, and model promotion automatically.
- **Streamlit Interface:** Interactive UI to predict accident risk using the trained model.

---

## ⚙️ Setup

1. **Clone the repository:**

```bash
git clone https://github.com/arkobera/PS5E10.git
cd PS5E10
````

2. **Create Python environment and install dependencies:**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file with:

```
CAPSTONE_TEST=<DagsHub token>
BUCKET_NAME=<S3 bucket name>
AWS_ACCESS_KEY=<AWS access key>
AWS_SECRET_KEY=<AWS secret key>
```

4. **DVC Setup (Optional if remote data exists):**

```bash
dvc pull
```

---

## 📊 Running the Pipeline

Reproduce the full pipeline:

```bash
dvc repro
```

This will run the following stages:

1. `data_ingestion` – Load raw data.
2. `Feature_Engineering` – Process features.
3. `model_building` – Train models.
4. `model_evaluation` – Evaluate model performance.
5. `model_registration` – Register and promote models.

---

## 🧪 Running Tests

Run all unit tests:

```bash
python -m unittest discover tests
```

---

## 💻 Streamlit App

Launch the Streamlit interface:

```bash
streamlit run WebInterface/app.py
```

* Provides interactive inputs for accident risk prediction.
* Displays processed features and model predictions.

---

## 📈 CI/CD

* GitHub Actions runs the pipeline on every push.
* Checks out the code, installs dependencies, runs `dvc repro`, tests, and promotes the model if successful.

---

## 📦 Requirements

* Python 3.10+
* `pandas`, `numpy`, `scipy`, `mlflow`, `dagshub`, `streamlit`, `python-dotenv`

---

## 🛠️ Notes

* Ensure all DVC remotes are configured correctly before running `dvc repro`.
* Make sure GitHub secrets are set for CI/CD (`CAPSTONE_TEST`, `BUCKET_NAME`, `AWS_ACCESS_KEY`, `AWS_SECRET_KEY`).

---

## 📜 License

MIT License



