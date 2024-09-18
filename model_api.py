from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)


# Load model and scaler
model = joblib.load('sponsor_roi_model.pkl')
scaler = joblib.load('scaler1.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        input_features = {
            'Event Type': data['event_type'],
            'Event Duration in Days': int(data['event_duration']),
            'Expected Footfall': (int(data['expected_max_footfall']) + int(data['expected_min_footfall'])) / 2,
            'Ticket Price': float(data['ticket_price']),
            'Sponsor Type': data['sponsor_type'],
            'Sponsor Cost': float(data['sponsor_cost'])
        }

        input_df = pd.DataFrame([input_features])

        predicted_revenue = model.predict(input_df)[0]
        sponsor_cost = input_df['Sponsor Cost'].values[0]
        roi = ((predicted_revenue - sponsor_cost) / sponsor_cost) * 100
        scaled_roi = scaler.transform([[roi]])[0][0]
        return jsonify({"predicted_revenue": float(predicted_revenue), "roi": float(roi), "scaled_roi": float(scaled_roi)})
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400


# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
