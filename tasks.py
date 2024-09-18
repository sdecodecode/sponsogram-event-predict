from celery import Celery
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from io import BytesIO
from PyPDF2 import PdfWriter, PdfReader
from reportlab.graphics.shapes import Drawing, Wedge, Polygon
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Frame, PageBreak
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import requests
import tempfile


load_dotenv()

model_api = 'https://sponsogram-event-predict-model.onrender.com/predict'

# Initialize Celery with environment variable for Redis
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
celery = Celery('tasks', broker=REDIS_URL, backend=REDIS_URL)

# Load model and scaler
# MODEL_PATH = os.getenv('MODEL_PATH', 'sponsor_roi_model.pkl')
# SCALER_PATH = os.getenv('SCALER_PATH', 'scaler1.pkl')


# try:
#     model = pickle.load(urllib.request.urlopen("https://drive.google.com/file/d/1NnyxG0srXl3QVpfIbLI_Pd_reCbh0HcT"))
#     scaler = pickle.load(urllib.request.urlopen("https://drive.google.com/file/d/188jNvCnA0QM8VZuHqwdFIQEUQQ7uWq_1"))
#     # logger.info("Model and scaler loaded successfully.")
# except Exception as e:
#     # logger.error(f"Error loading model.scaler: {e}")
#     raise e

# @celery.task
# def predict_roi(input_data):
#     try:
#         predicted_revenue = model.predict(input_data)[0]
#         sponsor_cost = input_data['Sponsor Cost'].values[0]
#         roi = ((predicted_revenue - sponsor_cost) / sponsor_cost) * 100
#         scaled_roi = scaler.transform([[roi]])[0][0]
#         return predicted_revenue, roi, scaled_roi
#     except Exception as e:
#         # logger.error(f"Error in predict_roi: {e}")
#         return None, None, None


@celery.task
def categorize_roi(scaled_roi):
    if scaled_roi <= 0.208755:
        return "Poor"
    elif scaled_roi <= 0.573525:
        return "Below Average"
    elif scaled_roi <= 1.392291:
        return "Average"
    elif scaled_roi <= 2.0:
        return "Good"
    else:
        return "Excellent"


@celery.task
def analyze_with_llama(prompt):
    try:
        client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
        )

        return (chat_completion.choices[0].message.content)

    except Exception as e:
        # logger.error(f"Error in Llama analysis: {e}")
        return f"An error occurred while querying the model: {e}"


TEAL = colors.Color(0.235, 0.561, 0.541)
BACKGROUND_GRAY = colors.Color(0.941, 0.941, 0.941)

styles = getSampleStyleSheet()


@celery.task
def create_gauge(roi_category, width, height):
    drawing = Drawing(width, height)
    categories = ['Excellent', 'Good', 'Average', 'Below Average', 'Poor']
    category_colors = [colors.green, colors.limegreen,
                       colors.yellow, colors.orange, colors.red]
    category_angles = [36, 72, 108, 144, 180]

    for i, (color, end_angle) in enumerate(zip(category_colors, category_angles)):
        start_angle = 0 if i == 0 else category_angles[i-1]
        section = Wedge(width / 2, height / 2, width / 2,
                        start_angle, end_angle, fillColor=color)
        drawing.add(section)

    angle = category_angles[categories.index(roi_category)] - 18
    needle_length = width / 2 - 20
    needle_end_x = width / 2 + needle_length * math.cos(math.radians(angle))
    needle_end_y = height / 2 + needle_length * math.sin(math.radians(angle))
    needle = Polygon(points=[width / 2, height / 2, width / 2 - 5, height / 2 + 10,
                             needle_end_x, needle_end_y, width / 2 + 5, height / 2 + 10],
                     fillColor=colors.black)
    drawing.add(needle)
    return drawing


@celery.task
def create_report_cover(buffer):
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    teal = colors.Color(0.235, 0.561, 0.541)
    c.setFillColor(teal)
    path = c.beginPath()
    path.moveTo(width-1.5*inch, 0.5*inch)
    path.lineTo(width-0.5*inch, 0.5*inch)
    path.lineTo(width-0.5*inch, 1.5*inch)
    path.lineTo(width-1.5*inch, 0.5*inch)
    c.drawPath(path, fill=1, stroke=0)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(1.3*inch, height-0.9*inch, "Sponsogram")
    c.setFont("Helvetica", 12)
    c.drawString(1.3*inch, height-1.1*inch, "Connect, Collaborate, Sponsor")

    logo_width = 0.6*inch
    logo_height = 0.6*inch
    logo_x = 1.3*inch - logo_width - 0.1*inch
    logo_y = height - 1.1*inch
    c.drawImage("logo.jpg", logo_x, logo_y,
                width=logo_width, height=logo_height)

    title_style = ParagraphStyle(
        'Title',
        fontName='Helvetica-Bold',
        fontSize=60,
        leading=68,
        textColor=colors.black,
    )

    formatted_date = datetime.now().strftime('%d/%m/%Y')

    title = "RETURN ON<br/>INVESTMENT<br/>ANALYSIS<br/>REPORT"
    p = Paragraph(title, title_style)

    frame = Frame(0.5*inch, 2*inch, width-inch, height-4*inch, showBoundary=0)
    frame.addFromList([p], c)

    c.setFont("Helvetica", 14)
    c.drawString(0.5*inch, 1.5*inch, f"Date: {formatted_date}")

    c.save()


title_style = ParagraphStyle(
    'Title', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=36, textColor=TEAL, spaceAfter=20, leading=40)
normal_style = ParagraphStyle(
    'Normal', parent=styles['Normal'], fontName='Helvetica', fontSize=12, leading=14)
user_input_style = ParagraphStyle(
    'UserInput', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=14, leading=14)


@celery.task
def generate_content_pages(buffer, plot1_img_path, event_type, event_duration, expected_min_footfall, expected_max_footfall, ticket_price, sponsor_type, sponsor_cost, roi_category, llama_output):
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []

    story.append(Paragraph("USER INPUTS", title_style))
    story.append(Spacer(1, 0.5 * inch))

    inputs = [
        f"Event Type: {event_type}",
        f"Event Duration: {event_duration}",
        f"Expected Min Footfall: {expected_min_footfall}",
        f"Expected Max Footfall: {expected_max_footfall}",
        f"Ticket Price: Rs. {ticket_price}",
        f"Sponsor Type: {sponsor_type}",
        f"Sponsor Cost: Rs. {sponsor_cost}",
        f"ROI Category: {roi_category}"
    ]

    for input_text in inputs:
        story.append(Paragraph(input_text, user_input_style))
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())
    story.append(Paragraph("DATA ANALYSIS PLOT", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Image(plot1_img_path, width=500, height=250))

    story.append(PageBreak())
    story.append(Paragraph("ROI CALCULATIONS", title_style))

    gauge_drawing = create_gauge(roi_category, 300, 200)
    story.append(Spacer(10, inch))
    story.append(gauge_drawing)

    str_roi = [
        "The expected ROI for the Sponsor is calculated based on the costs incurred and the revenue generated.",
        f"With an investment of Rs. {sponsor_cost} in sponsorship and advertising, the expected revenue is estimated to be '{roi_category}'."
    ]

    for line in str_roi:
        story.append(Paragraph(line, normal_style))

    story.append(PageBreak())
    story.append(Paragraph("SUMMARY", title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Process and add Llama output
    llama_lines = llama_output.split('\n')

    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=title_style,
        fontSize=16,
        leading=20,
        spaceAfter=6
    )
    subsubtitle_style = ParagraphStyle(
        'Subsubtitle',
        parent=subtitle_style,
        fontSize=14,
        leading=18,
        spaceAfter=4
    )

    for line in llama_lines:
        line = line.strip()
        if line.startswith('#'):
            header_level = len(line.split()[0])
            header_text = ' '.join(line.split()[1:])
            if header_level == 1:
                story.append(Paragraph(header_text, title_style))
            elif header_level == 2:
                story.append(Paragraph(header_text, subtitle_style))
            else:
                story.append(Paragraph(header_text, subsubtitle_style))
        elif line:
            story.append(Paragraph(line, normal_style))

        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)


@celery.task
def merge_pdfs(cover_buffer, content_buffer):
    pdf_writer = PdfWriter()

    cover_reader = PdfReader(cover_buffer)
    pdf_writer.add_page(cover_reader.pages[0])

    content_reader = PdfReader(content_buffer)
    for page in content_reader.pages:
        pdf_writer.add_page(page)

    merged_buffer = BytesIO()
    pdf_writer.write(merged_buffer)
    merged_buffer.seek(0)
    return merged_buffer


@celery.task
def generate_roi_plot(data, input_df, scaled_roi):
    sponsor_cost_range = np.arange(max(float(
        data['sponsor_cost']) - 100000, 10000), float(data['sponsor_cost']) + 100000, 10000)
    roi_categories_cost = []

    for cost in sponsor_cost_range:
        temp_input = input_df.copy()
        temp_input['Sponsor Cost'] = cost
        # _, _, temp_scaled_roi = predict_roi(temp_input)
        roi_categories_cost.append(categorize_roi(scaled_roi))

    category_order = ["Poor", "Below Average", "Average", "Good", "Excellent"]
    category_labels = sorted(set(roi_categories_cost),
                             key=lambda x: category_order.index(x))
    category_to_num = {cat: i for i, cat in enumerate(category_labels)}
    y_numeric = [category_to_num[cat] for cat in roi_categories_cost]

    plt.figure(figsize=(10, 5))
    plt.plot(sponsor_cost_range, y_numeric, marker='o')
    for i, category in enumerate(category_labels):
        plt.hlines(y=i, xmin=sponsor_cost_range[0], xmax=sponsor_cost_range[-1],
                   colors='blue', linestyles='dashed', alpha=0.5)
    plt.title("ROI Category vs Sponsor Cost")
    plt.xlabel("Sponsor Cost")
    plt.ylabel("ROI Category")
    plt.yticks(list(category_to_num.values()), category_labels)
    plt.xticks(sponsor_cost_range, rotation=45)
    plt.grid()

    plot_img_path = tempfile.mktemp(suffix=".png")
    plt.savefig(plot_img_path, format='png')
    plt.close()

    return plot_img_path


@celery.task
def generate_pdf(data, roi, scaled_roi, roi_category):
    prompt = (
        f"Given the following event details:\n"
        f"Event Type: {data['event_type']}\n"
        f"Event Duration: {data['event_duration']} days\n"
        f"Expected Footfall: {data['expected_min_footfall']} to {data['expected_max_footfall']}\n"
        f"Ticket Price: {data['ticket_price']} INR\n"
        f"Sponsor Type: {data['sponsor_type']}\n"
        f"Sponsor Cost: {data['sponsor_cost']} INR\n"
        f"ROI Category: {roi_category}\n\n"
        "Analyze the above information and recommend strategies to increase ROI from sponsor's point of view."
    )
    input_features = {
        'Event Type': data['event_type'],
        'Event Duration in Days': int(data['event_duration']),
        'Expected Footfall': (int(data['expected_max_footfall']) + int(data['expected_min_footfall'])) / 2,
        'Ticket Price': float(data['ticket_price']),
        'Sponsor Type': data['sponsor_type'],
        'Sponsor Cost': float(data['sponsor_cost'])
    }
    input_df = pd.DataFrame([input_features])
    # logger.debug("Calling Llama model")
    llama_output = analyze_with_llama(prompt)
    # logger.debug(f"Llama output: {llama_output}")

    # Generate ROI plot
    plot_img_path = generate_roi_plot(data, input_df, scaled_roi)

    # Generate PDF
    # logger.debug("Generating PDF")
    cover_buffer = BytesIO()
    create_report_cover(cover_buffer)

    content_buffer = BytesIO()
    generate_content_pages(content_buffer, plot_img_path, input_features['Event Type'], input_features['Event Duration in Days'],
                           data['expected_min_footfall'], data['expected_max_footfall'], input_features['Ticket Price'],
                           input_features['Sponsor Type'], input_features['Sponsor Cost'], roi_category, llama_output)

    merged_buffer = merge_pdfs(cover_buffer, content_buffer)

    # os.remove(plot_img_path)

    return llama_output, merged_buffer


@celery.task
def make_prediction(data):
    try:
        input_features = {
            'event_type': data['event_type'],
            'event_duration': int(data['event_duration']),
            'expected_min_footfall': int(data['expected_min_footfall']),
            'expected_max_footfall': int(data['expected_max_footfall']),
            'ticket_price': float(data['ticket_price']),
            'sponsor_type': data['sponsor_type'],
            'sponsor_cost': float(data['sponsor_cost'])
        }

        try:
            response = requests.post(model_api, json=input_features)
            response.raise_for_status()
            prediction_data = response.json()

            predicted_revenue = prediction_data['predicted_revenue']
            roi = prediction_data['roi']
            scaled_roi = prediction_data['scaled_roi']

            roi_category = categorize_roi(scaled_roi)

            # Handle any errors in the API call
        except requests.exceptions.RequestException as e:
            return {'error': f'API call failed: {str(e)}'}

        roi_category = categorize_roi(roi)

        llama_output, merged_buffer = generate_pdf(
            data, roi, scaled_roi, roi_category)

        result = {
            'prediction_revenue': float(predicted_revenue),
            'roi': float(roi),
            'scaled_roi': float(scaled_roi),
            'roi_category': roi_category,
            'event_type': data['event_type'],
            'event_duration': int(data['event_duration']),
            'expected_min_footfall': int(data['expected_min_footfall']),
            'expected_max_footfall': int(data['expected_max_footfall']),
            'ticket_price': float(data['ticket_price']),
            'sponsor_type': data['sponsor_type'],
            'sponsor_cost': float(data['sponsor_cost']),
            'llama_output': llama_output,
            # 'pdf_report': merged_buffer.getvalue()
        }
        # logger.debug(f"Prediction result: {result}")
        return result

    except Exception as e:
        # logger.error(f"Error in make_prediction: {str(e)}")
        # logger.error(traceback.format_exc())
        return {'error': str(e)}
