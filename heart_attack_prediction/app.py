import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join('models', 'best_model.pkl')
scaler_path = os.path.join('models', 'scaler.pkl')

print("Loading Machine Learning model and scaler...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# The EXACT feature order our Neural Network model expects
EXPECTED_COLS = ['age', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 
                 'sex_1', 'cp_1', 'cp_2', 'cp_3', 'fbs_1', 'restecg_1', 'restecg_2']

# Only the numerical columns were scaled during our training process
NUM_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Fetch form data safely and convert to float/int
            age = float(request.form.get('age', 0))
            sex = int(request.form.get('sex', 0))
            cp = int(request.form.get('cp', 0))
            trestbps = float(request.form.get('trestbps', 0))
            chol = float(request.form.get('chol', 0))
            fbs = int(request.form.get('fbs', 0))
            restecg = int(request.form.get('restecg', 0))
            thalach = float(request.form.get('thalach', 0))
            exang = int(request.form.get('exang', 0))
            oldpeak = float(request.form.get('oldpeak', 0.0))
            slope = int(request.form.get('slope', 0))
            ca = float(request.form.get('ca', 0))
            thal = int(request.form.get('thal', 0))

            # 2. Map form inputs to the 16 exact identical features the model expects
            # This logic basically perfectly replicates 'One-Hot Encoding'
            features = {
                'age': age,
                'trestbps': trestbps,
                'chol': chol,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal,
                # Create One-Hot Dummy variables on the fly
                'sex_1': 1 if sex == 1 else 0,
                'cp_1': 1 if cp == 1 else 0,
                'cp_2': 1 if cp == 2 else 0,
                'cp_3': 1 if cp == 3 else 0,
                'fbs_1': 1 if fbs == 1 else 0,
                'restecg_1': 1 if restecg == 1 else 0,
                'restecg_2': 1 if restecg == 2 else 0,
            }

            # Convert our dictionary to a pandas DataFrame with the EXACT expected column order
            input_df = pd.DataFrame([features])[EXPECTED_COLS]

            # 3. Apply the exact same Scaling transformation
            input_df[NUM_COLS] = scaler.transform(input_df[NUM_COLS])

            # 4. Predict
            prediction = model.predict(input_df)[0]
            
            # 5. Format Output safely using the user's required wording
            if prediction == 1:
                result_text = "⚠️ High Risk of Heart Attack detected. Please consult a doctor."
                result_class = "danger"
            else:
                result_text = "✅ Low Risk of Heart Attack. Maintain a healthy lifestyle."
                result_class = "success"

            explanation = "This prediction is based on the data you entered and a machine learning model."

            return render_template('index.html', result_text=result_text, result_class=result_class, explanation=explanation)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred: {str(e)}")

# This ensures it can be deployed correctly on Render or locally
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
