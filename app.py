from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("carbon_model.pkl")

FEATURES = [
    "Vehicle Monthly Distance Km",
    "Frequency of Traveling by Air",
    "Vehicle Type",
    "How Many New Clothes Monthly",
    "Waste Bag Weekly Count",
    "Heating Energy Source"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Keys now match exactly what the HTML sends
        input_dict = {
            "Vehicle Monthly Distance Km": float(data["vehicle_km"]),
            "Frequency of Traveling by Air": str(data["air_travel"]),
            "Vehicle Type":                  str(data["vehicle_type"]),
            "How Many New Clothes Monthly":  float(data["new_clothes"]),
            "Waste Bag Weekly Count":        float(data["waste_bags"]),
            "Heating Energy Source":         str(data["heating_source"]),
        }

        features = pd.DataFrame([input_dict], columns=FEATURES)
        prediction = model.predict(features)[0]
        

        return jsonify({
            "user_emissions": round(float(prediction), 2), "us_average": 4000})

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)