import mlflow
import os
import sys

def get_latest_run_id():
    
    experiment_name = "Loan_Approval_Classification_Advance"
    
    try:
        # Cari eksperimen berdasarkan nama
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print("Eksperimen dengan nama tersebut tidak ditemukan, mencoba ID 0...")
            experiment = mlflow.get_experiment("0")
            
        if experiment is None:
            print(f"Error: Eksperimen '{experiment_name}' atau ID 0 tidak ditemukan.")
            sys.exit(1)
        
        experiment_id = experiment.experiment_id
        print(f"Menggunakan Experiment ID: {experiment_id}")
        
        # Cari run terakhir yang statusnya FINISHED
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            print(f"Error: Tidak ada run yang sukses (FINISHED) di eksperimen ID {experiment_id}.")
            sys.exit(1)
            
        # Ambil Run ID
        run_id = runs.iloc[0].run_id
        
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        print(f"Berhasil mendapatkan Run ID: {run_id}")
        
    except Exception as e:
        print(f"Terjadi kesalahan saat mengambil Run ID: {e}")
        sys.exit(1)

if __name__ == "__main__":
    get_latest_run_id()