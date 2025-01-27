# Save this as app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "Welcome to the ML Model API!"



@app.route('/predict', methods=['GET'])
def predict():
    try:
        
        # Parse JSON request
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)