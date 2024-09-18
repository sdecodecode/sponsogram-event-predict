from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app) 


model = joblib.load('sponsor_roi_model.pkl')
scaler = joblib.load('scaler1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_features = {
            'Event Type': data['event_type'],
            'Event Duration in Days': int(data['event_duration']),
            'Expected Footfall': (int(data['expected_max_footfall']) + int(data['expected_min_footfall'])) / 2,
            'Ticket Price': float(data['ticket_price']),
            'Sponsor Type': data['sponsor_type'],
            'Sponsor Cost': float(data['sponsor_cost'])
        }
        # logger.debug(f"Input features: {input_features}")
    input_df = pd.DataFrame([input_features])   
    # event_type = data['event_type']
    # event_duration = int(data['event_duration']),
    # expected_footfall = (int(data['expected_max_footfall']) + int(data['expected_min_footfall'])) / 2,
    # ticket_price = float(data['ticket_price']),
    # sponsor_type = data['sponsor_type'],
    # sponsor_cost = float(data['sponsor_cost'])

    # input_features = np.array([[event_type, event_duration, expected_footfall, ticket_price, sponsor_type, sponsor_cost]])

    predicted_revenue = model.predict(input_df)[0]
    sponsor_cost = input_df['Sponsor Cost'].values[0]
    roi = ((predicted_revenue - sponsor_cost) / sponsor_cost) * 100
    scaled_roi = scaler.transform([[roi]])[0][0]
    return jsonify({"predicted_revenue": float(predicted_revenue), "roi": float(roi), "scaled_roi": float(scaled_roi)})

@app.route('/test', methods=['GET'])
def api_test():
    response_data = {"message": "Model API is working :)!"}
    return jsonify(response_data), 200 
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
