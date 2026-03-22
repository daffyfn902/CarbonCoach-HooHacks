from flask import Flask, render_template, request, jsonify
from google import genai
from dotenv import load_dotenv
import joblib
import pandas as pd
import os


load_dotenv()


app = Flask(__name__)


# ── Load model + feature importances ────────────────────────────────────────
model = joblib.load("carbon_model.pkl")
feature_importance = joblib.load("feature_importance.pkl")


# ── Configure Gemini client ──────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


FEATURES = [
    "Vehicle Monthly Distance Km",
    "Frequency of Traveling by Air",
    "Vehicle Type",
    "How Many New Clothes Monthly",
    "Waste Bag Weekly Count",
    "Heating Energy Source"
]


US_AVERAGE_KG = 4000




@app.route("/")
def home():
    return render_template("index.html")




@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()


        vehicle_type = None if data["vehicle_type"] == "none" else str(data["vehicle_type"])


        input_dict = {
            "Vehicle Monthly Distance Km": float(data["vehicle_km"]),
            "Frequency of Traveling by Air": str(data["air_travel"]),
            "Vehicle Type":                  vehicle_type,
            "How Many New Clothes Monthly":  float(data["new_clothes"]),
            "Waste Bag Weekly Count":        float(data["waste_bags"]),
            "Heating Energy Source":         str(data["heating_source"]),
        }


        features = pd.DataFrame([input_dict], columns=FEATURES)
        prediction = model.predict(features)[0]


        # ── Per-user contribution mapping ────────────────────────────────────
        air_key     = f"Frequency of Traveling by Air_{data['air_travel']}"
        vehicle_key = f"Vehicle Type_{vehicle_type}" if vehicle_type else "Vehicle Type_nan"
        heat_key    = f"Heating Energy Source_{data['heating_source']}"


        user_contributions = {
            "Monthly Driving":  feature_importance.get("Vehicle Monthly Distance Km", 0),
            "Air Travel":       feature_importance.get(air_key, 0),
            "Vehicle Type":     feature_importance.get(vehicle_key, 0),
            "New Clothes":      feature_importance.get("How Many New Clothes Monthly", 0),
            "Waste Production": feature_importance.get("Waste Bag Weekly Count", 0),
            "Heating Source":   feature_importance.get(heat_key, 0),
        }


        def by_importance(item):
            return item[1]


        ranked_contributors = sorted(
            user_contributions.items(), key=by_importance, reverse=True
        )


        return jsonify({
            "user_emissions":      round(float(prediction), 2),
            "us_average":          US_AVERAGE_KG,
            "ranked_contributors": ranked_contributors,
            "top_contributor":     ranked_contributors[0][0],
        })


    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/coach", methods=["POST"])
def coach():
    try:
        data = request.get_json()


        kg           = data["prediction_kg"]
        us_avg       = data["us_average"]
        contributors = data["ranked_contributors"]


        # ── Translate dropdown values to readable labels ──────────────────────
        drive_map = {
            "1000": "low (under 1,250 miles/month)",
            "4000": "moderate (1,250–3,700 miles/month)",
            "9000": "high (3,700+ miles/month)"
        }
        air_map = {
            "rarely":          "rarely (0–2 flights/year)",
            "frequently":      "sometimes (3–5 flights/year)",
            "very frequently": "often (6+ flights/year)"
        }
        clothes_map = {
            "1": "low (0–2 items/month)",
            "4": "moderate (3–5 items/month)",
            "7": "high (6+ items/month)"
        }
        waste_map = {
            "1": "low (1 bag/week)",
            "2": "moderate (2–3 bags/week)",
            "4": "high (4+ bags/week)"
        }


        drive_label   = drive_map.get(str(data["vehicle_km"]),    str(data["vehicle_km"]))
        air_label     = air_map.get(data["air_travel"],           data["air_travel"])
        clothes_label = clothes_map.get(str(data["new_clothes"]), str(data["new_clothes"]))
        waste_label   = waste_map.get(str(data["waste_bags"]),    str(data["waste_bags"]))
        vehicle_label = "no car" if data["vehicle_type"] == "none" else data["vehicle_type"]


        contributors_text = "\n".join(
            f"  - {name}: {round(weight * 100, 1)}% contribution"
            for name, weight in contributors
        )


        percent_vs_avg = round((kg / us_avg) * 100)


        system_prompt = f"""
You are an environmental sustainability coach. Be concise, warm, and specific.


A user completed a carbon footprint survey with these lifestyle habits:
- Monthly driving: {drive_label}
- Vehicle type: {vehicle_label}
- Air travel: {air_label}
- New clothes per month: {clothes_label}
- Weekly waste: {waste_label}
- Heating source: {data["heating_source"]}


Their estimated carbon footprint is {kg} kg CO2/year.
The average is {us_avg} kg/year.
They are at {percent_vs_avg}% of the average.


Based on the model, here is how much each factor contributed to their specific footprint:
{contributors_text}
"""


        # ── Create chat session with system prompt ───────────────────────────
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config={
                "system_instruction": system_prompt
            }
        )


        message = """
Respond in exactly 3 short paragraphs with no headers or bold text:
1. What their footprint means relative to average (1-2 sentences, be encouraging if low)
2. Their single biggest contributor from the list above and why it matters (2-3 sentences)
3. Three specific, actionable ways to reduce their footprint prioritized by biggest contributors — use bullet points starting with •


Do not use jargon. Do not repeat the numbers more than once. Keep the total response under 150 words.
"""


        response = chat.send_message(message)
        return jsonify({"coach_response": response.text})


    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)




