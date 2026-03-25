import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Handle missing values
    if df.isnull().sum().sum() > 0:
        print("Handling missing values by dropping them...")
        df.dropna(inplace=True)
        
    print(f"Dataset Shape after missing values handling: {df.shape}")
    
    # 2. Identify categorical and numerical columns
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    num_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    num_cols = [col for col in num_cols if col in df.columns]
    
    target_col = 'output'
    if target_col not in df.columns:
        if 'target' in df.columns:
            target_col = 'target'
        else:
            target_col = df.columns[-1]
            
    print(f"Target column detected as: {target_col}")

    # 3. Encode categorical variables using One-Hot Encoding
    print("Encoding categorical variables using One-Hot Encoding...")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # 4. Split data into Features (X) and Target (y)
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    # 5. Train-Test Split (80-20)
    print("Splitting data into 80% train and 20% test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 6. Normalize/Standardize numerical data
    print("Standardizing numerical features...")
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # We only apply scaling to the original numerical columns, properly preserving One-Hot columns
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns
