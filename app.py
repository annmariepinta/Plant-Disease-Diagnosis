import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load trained model
model = load_model('plant_disease_model.h5')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Ensure the uploads directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict class
        predicted_label, confidence = predict_image_class(file_path)

        return render_template('result.html', filename=filename, predicted_label=predicted_label, confidence=confidence)

    return redirect(request.url)

def predict_image_class(image_path):
    # Load image and convert to RGB format
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((150, 150))  # Resize image to match model's expected sizing
    img_array = np.array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # Get class labels
    class_labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}
    predicted_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class]

    return predicted_label, confidence

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
