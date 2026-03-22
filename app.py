from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model pipeline
model = joblib.load("carbon_model.pkl")

# Define the feature names exactly as in the reduced model
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
    data = request.get_json()

    # Read inputs from frontend JSON
    vehicle_km = float(data.get("miles", 0)) 
    flights = float(data.get("flights", 0))
    vehicle_type = data.get("vehicle", "")
    clothes = float(data.get("clothes", 0))
    waste = float(data.get("waste", 0))
    heating = data.get("heating", "")

    # Build a DataFrame with the same columns
    input_dict = {
        "Vehicle Monthly Distance Km": vehicle_km,
        "Frequency of Traveling by Air": flights,
        "Vehicle Type": vehicle_type,
        "How Many New Clothes Monthly": clothes,
        "Waste Bag Weekly Count": waste,
        "Heating Energy Source": heating
    }

    features = pd.DataFrame([input_dict], columns=FEATURES)

    # Predict
    prediction = model.predict(features)[0]

    return jsonify({
        "user_emissions": round(prediction, 2),
        "us_average": round(2269, 2)
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)