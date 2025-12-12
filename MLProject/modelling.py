import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def train_and_log_model():
    print("Memulai proses training (Lokal)...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("loan_data_cleaned_automated.csv")
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan.")
        return

    # 2. Split Data
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow 
    mlflow.set_experiment("Loan_Approval_Local_CI")

    with mlflow.start_run(run_name="RandomForest_CI_Run") as run:
        
        # --- HYPERPARAMETERS ---
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log Params
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # --- METRICS & PREDICT ---
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        
        # --- ARTIFACTS ---
        with open("best_parameters.json", "w") as f:
            json.dump(best_params, f)
        mlflow.log_artifact("best_parameters.json")

        # --- LOG MODEL ---
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        # --- SIMPAN RUN ID ---
        run_id = run.info.run_id
        print(f"Saving Run ID: {run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print("Training selesai. Artefak tersimpan di folder ./mlruns lokal.")

if __name__ == "__main__":
    train_and_log_model()