import logging
from flask import Flask, request, jsonify, make_response
from tasks import make_prediction
from celery import Celery

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG) 

celery = Celery('tasks',
    backend='redis://localhost:6379/0',
    broker='redis://localhost:6379/0')
celery.conf.update(app.config)

TASKS = {}

@celery.task
def make_prediction_task(data):
    return make_prediction(data)

def create_task(data):
    task = make_prediction_task.delay(data)
    task_id = task.id
    result = make_prediction(data) 
    TASKS[task_id] = {
        'event_type': data['event_type'],
        'event_duration': data['event_duration'],
        'expected_min_footfall': data['expected_min_footfall'],
        'expected_max_footfall': data['expected_max_footfall'],
        'ticket_price': data['ticket_price'],
        'sponsor_type': data['sponsor_type'],
        'sponsor_cost': data['sponsor_cost'],
        'prediction_revenue': result['prediction_revenue'],
        'roi': result['roi'],
        'scaled_roi': result['scaled_roi'],
        'roi_category': result['roi_category'],
        'llama_output': result['llama_output'],
        'pdf_report': result.get('pdf_report', None),
        'task_id': task_id,
    }
    return task_id

def validate_input(data):
    required_fields = ['event_type', 'event_duration', 'expected_min_footfall', 'expected_max_footfall', 'ticket_price', 'sponsor_type', 'sponsor_cost']
    
    for field in required_fields:
        if field not in data or data[field] is None:
            return False, f'Missing field: {field}'
    
    try:
        int(data['event_duration'])
        int(data['expected_min_footfall'])
        int(data['expected_max_footfall'])
        float(data['ticket_price'])
        float(data['sponsor_cost'])
    except ValueError:
        return False, 'Invalid type for one of the numeric fields'
    
    return True, None

@app.route('/get_id', methods=['GET'])
def predict():
    try:
        app.logger.info("Received request")
        data = request.args.to_dict()
        app.logger.info(f"Received data: {data}")

        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        task_id = create_task(data)
        return jsonify({'task_id': task_id}), 202

    except Exception as e:
        app.logger.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['GET'])
def download_pdf():
    task_id = request.args.get('task_id')
    if task_id in TASKS:
        result = TASKS[task_id]
        pdf_data = result.get('pdf_report', None)  
        if pdf_data:
            response = make_response()
            response.data = pdf_data  
            response.headers['Content-Type'] = 'application/pdf'
            response.headers['Content-Disposition'] = f'attachment; filename=report_{task_id}.pdf'
            return response
            
        else:
            return jsonify({'error': 'PDF report not found for this task'}), 404
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/predict', methods=['GET'])
def get_task():
    task_id = request.args.get('task_id')
    if task_id in TASKS:
        result = TASKS[task_id]
        result.pop('pdf_report', None)
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Task not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)