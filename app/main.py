from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("iris_model.pkl")
target_names = ['setosa', 'versicolor', 'virginica']

@app.route("/")
def home():
    return "Welcome to Iris Predictor API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return jsonify({"prediction": target_names[prediction]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
