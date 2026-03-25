import os
import pandas as pd
from src.preprocessing import load_and_preprocess_data
from src.eda import perform_eda
from src.models import train_and_evaluate_models, save_best_model

def main():
    print("==================================================")
    print("   Heart Attack Risk Prediction ML Pipeline       ")
    print("==================================================")
    
    data_path = 'data/heart.csv'
    
    if not os.path.exists(data_path):
        print(f"\n[ERROR] Dataset not found at {data_path}.")
        print("Please download the 'Heart Attack Analysis & Prediction Dataset' from Kaggle:")
        print("URL: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset")
        print("And place the 'heart.csv' file inside the 'data/' folder.")
        return

    # Step 1: Exploratory Data Analysis
    print("\n--- Step 1: Exploratory Data Analysis (EDA) ---")
    df = pd.read_csv(data_path)
    perform_eda(df, output_dir='outputs')
    
    # Step 2: Data Preprocessing
    print("\n--- Step 2: Data Preprocessing ---")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data(data_path)
    print("Data preprocessing complete.")

    # Step 3: Train and Evaluate Models
    print("\n--- Step 3: Model Training & Evaluation ---")
    best_model, best_model_name, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_names, output_dir='outputs'
    )

    # Step 4: Save Best Model
    print("\n--- Step 4: Saving Best Model ---")
    save_best_model(best_model, scaler, models_dir='models')
    
    print("\n==================================================")
    print("   Pipeline Execution Finished Successfully!      ")
    print("==================================================")
    print("Check 'outputs/' folder for ROC curve, Feature Importance, and Confusion Matrix.")
    print("Check 'models/' folder for the serialized best model ('best_model.pkl') and scaler ('scaler.pkl').")

if __name__ == "__main__":
    main()
