from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form or request.get_json() or {}
    try:
        features = [
            float(data.get("sepal_length")),
            float(data.get("sepal_width")),
            float(data.get("petal_length")),
            float(data.get("petal_width"))
        ]
    except Exception:
        return jsonify({"error": "Please provide 4 numeric values"}), 400

    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)[0]
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
