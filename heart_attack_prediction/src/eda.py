import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='outputs'):
    print("Performing Exploratory Data Analysis...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    target_col = 'output' if 'output' in df.columns else ('target' if 'target' in df.columns else df.columns[-1])

    # 1. Target Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Target Variable Distribution (0 = Low Risk, 1 = High Risk)')
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 8))
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 3. Histograms and Boxplots for numerical columns
    num_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    num_cols = [col for col in num_cols if col in df.columns]
    
    for col in num_cols:
        plt.figure(figsize=(12, 5))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Histogram of {col}')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[col], x=df[target_col], palette='Set2')
        plt.title(f'Boxplot of {col} by Target')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'))
        plt.close()
        
    print(f"EDA visualisations saved to '{output_dir}/'")
