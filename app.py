from celery.result import AsyncResult
from flask import Flask, request, jsonify
from flask_cors import CORS
from tasks import make_prediction, generate_pdf, celery
import base64
import os
import redis

app = Flask(__name__)
CORS(app)

# Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Initialize Redis client
# Adjust host and port if your Redis server is running elsewhere
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.Redis.from_url(REDIS_URL)


def validate_input(data):
    required_fields = [
        'event_type', 'event_duration', 'expected_min_footfall',
        'expected_max_footfall', 'ticket_price', 'sponsor_type',
        'sponsor_cost'
    ]

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


@app.route('/predict', methods=['GET'])
def get_prediction():
    try:
        # logger.info("Received prediction request")
        data = request.args.to_dict()

        # Synchronous validation
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        task = make_prediction.apply_async(args=[data])
        task_id = task.id

        # Optionally, store the task_id in a separate Redis set for tracking
        redis_client.sadd('known_tasks', task_id)

        return jsonify({'task_id': task_id}), 202
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    # logger.info("Received status request for Task ID: %s", task_id)

    # Check if the task_id exists in the 'known_tasks' set
    task_exists = redis_client.sismember('known_tasks', task_id)

    if not task_exists:
        return jsonify({'state': 'UNKNOWN', 'error': 'Task ID does not exist.'}), 404

    task = AsyncResult(task_id, app=celery)
    # logger.info("Task State: %s", task.state)

    if task.state == 'PENDING':
        return jsonify({'state': task.state, 'message': 'Task is pending...'}), 202
    elif task.state == 'SUCCESS':
        return jsonify({
            'state': task.state,
            'result': task.result,
            'task_id': task.id
        }), 200
    elif task.state == 'FAILURE':
        return jsonify({'state': task.state, 'error': str(task.info)}), 500
    else:
        return jsonify({'state': task.state, 'message': 'Task is in progress.'}), 202


@app.route('/generate-report/<task_id>', methods=['GET'])
def generate_report(task_id):
    # logger.info("Received generate report request for Task ID: %s", task_id)

    # Check if the task_id exists in the 'known_tasks' set
    task_exists = redis_client.sismember('known_tasks', task_id)

    if not task_exists:
        return jsonify({'state': 'UNKNOWN', 'error': 'Task ID does not exist.'}), 404

    task = AsyncResult(task_id, app=celery)
    # logger.info("Task State: %s", task.state)

    if task.state == 'SUCCESS':
        result = task.result
        llama_output, merged_buffer = generate_pdf(
            result, result['roi'], result['scaled_roi'], result['roi_category'])

        if merged_buffer:
            try:
                # Option 1: Return PDF directly
                # pdf_buffer = merged_buffer
                # pdf_buffer.seek(0)  # Ensure buffer is at the beginning

                # return send_file(
                #     pdf_buffer,
                #     as_attachment=True,
                #     download_name='event_report.pdf',
                #     mimetype='application/pdf'
                # )

                # Option 2: Return PDF as Base64 (commented out)

                pdf_data = merged_buffer.getvalue()
                pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')

                response = {
                    'state': task.state,
                    'message': 'Report generated successfully.',
                    'pdf_data': pdf_base64  # Base64-encoded PDF
                }
                return jsonify(response), 200

            except Exception as e:
                # logger.exception("Error generating PDF")
                return jsonify({'error': f'Error generating PDF: {str(e)}'}), 500
        else:
            # logger.error("Merged buffer is empty.")
            return jsonify({'error': 'Failed to generate PDF.'}), 500

    elif task.state == 'FAILURE':
        # logger.error("Task Failure: %s", task.info)
        return jsonify({
            'state': task.state,
            'error': str(task.info)
        }), 500

    else:
        return jsonify({'state': task.state, 'message': 'Report is not ready yet.'}), 202


# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(debug=True)
