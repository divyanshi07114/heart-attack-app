import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Initializing models and defining hyperparameter grids...")
    
    # Define models and their parameter grids for GridSearchCV
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs']
        }),
        'Random Forest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }),
        'Support Vector Machine': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        }),
        'Neural Network': (MLPClassifier(max_iter=1000, random_state=42), {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.05]
        })
    }
    
    best_model = None
    best_model_name = ""
    best_f1 = -1
    model_results = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, (model, param_grid) in models.items():
        print(f"\nTraining and tuning {name}...")
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_estimator = grid.best_estimator_
        print(f"Best parameters for {name}: {grid.best_params_}")
        
        # Predictions
        y_pred = best_estimator.predict(X_test)
        
        # Predict Probabilities for ROC-AUC
        if hasattr(best_estimator, "predict_proba"):
            y_prob = best_estimator.predict_proba(X_test)[:, 1]
        else:
            y_prob = best_estimator.decision_function(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Metrics -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        model_results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'ROC-AUC': roc_auc}
        
        # Plot ROC Curve for this model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        
        # Identify the Best Model based on F1-Score
        if f1 > best_f1:
            best_f1 = f1
            best_model = best_estimator
            best_model_name = name
            best_y_pred = y_pred

    # Render ROC Curve
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    print(f"\n*** Best Model identified as: {best_model_name} ***")
    
    # 7. Confusion Matrix for Best Model
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, 'best_model_confusion_matrix.png'))
    plt.close()
    
    # 8. Feature Importance for Best Model
    if hasattr(best_model, "feature_importances_") or hasattr(best_model, "coef_"):
        plt.figure(figsize=(10, 6))
        
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        else:
            importances = np.abs(best_model.coef_[0])
            
        indices = np.argsort(importances)[::-1]
        
        plt.title(f"Feature Importances ({best_model_name})")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center", color='teal')
        plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
    else:
        print(f"Model {best_model_name} does not support straightforward feature importance extraction.")

    # Show Final Benchmark Summary
    print("\nModel Benchmark Summary:")
    df_results = pd.DataFrame(model_results).T
    print(df_results)
    
    return best_model, best_model_name, model_results

def save_best_model(model, scaler, models_dir='models'):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    model_path = os.path.join(models_dir, 'best_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
