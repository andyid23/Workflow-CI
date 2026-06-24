import flask, joblib, os
import numpy as np

app = flask.Flask(__name__)
model = joblib.load("model/rf_model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = flask.request.json
    df = np.array([data.get("features", data)])
    pred = model.predict(df)
    return flask.jsonify({"prediction": pred.tolist()})

@app.route("/health")
def health():
    return flask.jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
