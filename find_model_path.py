import os
import sys

def find_latest_model():
    start_dir = os.path.join(os.getcwd(), "MLProject", "mlruns")
    
    latest_time = 0
    latest_model_dir = None

    for root, dirs, files in os.walk(start_dir):
        if "model.pkl" in files:
            file_path = os.path.join(root, "model.pkl")
            file_time = os.path.getmtime(file_path)
            
            if file_time > latest_time:
                latest_time = file_time
                latest_model_dir = root

    if latest_model_dir:
        print(f"file://{latest_model_dir}")
    else:
        print("ERROR: Model not found")
        sys.exit(1)

if __name__ == "__main__":
    find_latest_model()