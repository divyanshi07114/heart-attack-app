# Heart Attack Risk Prediction Project

## Overview
This repository contains a complete, beginner-friendly machine learning pipeline for predicting heart attack (myocardial infarction) risk using Python. It uses the [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) from Kaggle. 

The project addresses a **Binary Classification** problem (0 = low risk, 1 = high risk). It features Exploratory Data Analysis (EDA), automatic data preprocessing (handling missing values, standardizing, one-hot encoding), checking different models using `GridSearchCV` hyperparameter tuning, and saving the best performing model.

## Features
- **Data Preprocessing:** Handle missing data, standardize numerical features, and encode categorical features.
- **Exploratory Data Analysis (EDA):** Automatic generation of correlation matrices, target distributions, and feature distributions.
- **Models Used:** Logistic Regression, Random Forest, Support Vector Machine (SVM), and Neural Network (MLP).
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Visualizations:** ROC curves, Confusion Matrix, and Feature Importance of the best model.
- **Model Persistence:** The best performing model and standard scaler are saved automatically using `pickle`.

## Project Structure
```text
heart_attack_prediction/
│
├── data/                   # Place the Kaggle dataset here
│   └── heart.csv           # (You need to download this securely from kaggle)
│
├── models/                 # Saved models and scalers will appear here
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── outputs/                # Generated visualizations are saved here
│   ├── best_model_confusion_matrix.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── roc_curves.png
│   └── target_distribution.png
│
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── eda.py              # Exploratory Data Analysis script
│   ├── models.py           # Model training and fine-tuning logic
│   └── preprocessing.py    # Data loading and transformation tools
│
├── main.py                 # The execution entry point
├── requirements.txt        # Required python dependencies
└── README.md               # Project documentation
```

## How to Run the Project

1. **Clone/Download the repository** manually onto your local machine.

2. **Download the dataset:**
   - Go to Kaggle: [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)
   - Download the data and place the file appropriately named `heart.csv` into the `data/` folder of this project.

3. **Install Dependencies:**
   - Make sure you have python installed. It is recommended to create a virtual environment:
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Linux/macOS
     venv\Scripts\activate     # On Windows
     ```
   - Run the command to install packages from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Model Pipeline:**
   - Run `main.py` directly from the main directory:
     ```bash
     python main.py
     ```
   - The script will systematically execute EDA, preprocess, perform hyperparameter tuning using Grid Search, output performance, and save the best model to the `models/` directory along with analytical plots to `outputs/`.

## Beginner's Explanation
- **Data Preprocessing:** Before feeding data to models, we scaled numerical numbers (so massive values like cholesterol don't overshadow small ones like age), created dummy variables for categorical data, and ensured there were no missing rows.
- **Support Vector Machine (SVM):** It draws a hyper-plane (a mathematical line) trying to separate low risk from high risk perfectly.
- **Random Forest:** Acts like a group of decision trees acting together to vote on the outcome. It usually handles complex logic well.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** A score representing how well our model can distinguish between high-risk and low-risk patients correctly regardless of where we place the threshold. Higher is better!
- **GridSearchCV:** It automatically tries out hundreds of combinations of model settings (hyperparameters) and finds the most optimal mathematical parameters for you by cross-validating the performance over 5-folds of the given data!

## Future Improvements
- Create a front-end UI (`Streamlit` or `Flask`) that loads the pickled model and predicts user's inputs interactively.
- Deploy the web UI on platforms like Render or Vercel.
