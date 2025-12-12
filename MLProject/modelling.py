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
import dagshub

# --- KONFIGURASI DAGSHUB ---
REPO_OWNER = "raflikamal" 
REPO_NAME = "SMSML_M-Rafli-Kamal" 

# Fungsi ini otomatis setup tracking URI & Autentikasi
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

# --- FUNGSI PELATIHAN & LOGGING ---
def train_and_log_model():
    print("Memulai proses training...")
    
    # 1. Load Data
    try:
        df = pd.read_csv("loan_data_cleaned_automated.csv")
    except FileNotFoundError:
        print("Error: File dataset 'loan_data_cleaned_automated.csv' tidak ditemukan di folder ini.")
        return

    # 2. Split Data
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow Experiment
    experiment_name = "Loan_Approval_Classification_Advance"
    try:
        mlflow.set_experiment(experiment_name)
    except:
        print(f"Eksperimen {experiment_name} mungkin sudah ada atau terjadi error saat pembuatan.")

    with mlflow.start_run(run_name="RandomForest_ManualLogging"):
        
        # --- HYPERPARAMETERS & GRID SEARCH (Opsional, untuk menghasilkan grid_search_results.csv) ---
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Ambil model terbaik
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # Log Best Parameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
            
        # Simpan hasil Grid Search ke CSV
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("grid_search_results.csv", index=False)
        mlflow.log_artifact("grid_search_results.csv")

        # Simpan Best Parameters ke JSON
        with open("best_parameters.json", "w") as f:
            json.dump(best_params, f)
        mlflow.log_artifact("best_parameters.json")

        # --- PREDIKSI & METRIK ---
        y_pred = best_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary') 
        rec = recall_score(y_test, y_pred, average='binary')     
        f1 = f1_score(y_test, y_pred, average='binary')         
        
        print(f"Accuracy: {acc}")
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1-Score: {f1}")

        # Log Metrics (Manual)
        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        mlflow.log_metrics(metrics)
        
        # Simpan Metrics Summary ke JSON
        with open("metrics_summary.json", "w") as f:
            json.dump(metrics, f)
        mlflow.log_artifact("metrics_summary.json")

        # Simpan Prediksi ke CSV
        predictions_df = X_test.copy()
        predictions_df['Actual'] = y_test
        predictions_df['Predicted'] = y_pred
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
        
        # --- ARTEFAK GAMBAR ---
        
        # 1. Confusion Matrix Plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png")
        
        # 2. Feature Importance Plot & CSV
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
            
            # Simpan plot
            plt.figure(figsize=(10, 6))
            feature_importances.nlargest(10).plot(kind='barh')
            plt.title('Top 10 Feature Importances')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            plt.close()
            mlflow.log_artifact("feature_importance.png")
            
            # Simpan data feature importance ke CSV
            feature_importances.sort_values(ascending=False).to_csv("feature_importance.csv")
            mlflow.log_artifact("feature_importance.csv")
        
        # Log Model (Manual)
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        print("Training selesai. Semua artefak (CSV, JSON, PNG, Model) tersimpan di DagsHub MLflow.")
        
        # Bersihkan file lokal sementara
        for file in ["confusion_matrix.png", "feature_importance.png", "grid_search_results.csv", 
                     "best_parameters.json", "metrics_summary.json", "predictions.csv", "feature_importance.csv"]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    train_and_log_model()