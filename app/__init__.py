# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,send_file
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
import os
import cv2
import numpy as np
import secrets
import time

class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileAllowed(['jpg', 'png', 'jpeg', 'gif','tiff'], 'Images only!')])
app = Flask(__name__)
app.config['SECRET_KEY'] =  secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['PROCESSED_FOLDER'] = 'app/static/processed'


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()  # create an instance of UploadForm
    if request.method == 'POST':
        if form.validate_on_submit():
            file = form.image.data
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            file.save(upload_path)
            return redirect(url_for('process', filename=filename))
    return render_template('index.html', form=form)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('process', filename=filename))
    return render_template('templates/upload.html')


@app.route('/process/<filename>', methods=['GET', 'POST'])
def process(filename):
    if request.method == 'POST':
        operation = request.form.get('operation')
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if operation == 'canny':
            edges = cv2.Canny(img, 100, 200)
            cv2.imwrite(output_path, edges)
        elif operation == 'hough':
            edges = cv2.Canny(img, 50, 150, apertureSize = 3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imwrite(output_path, img)
        elif operation == 'harris':
            img = np.float32(img)
            dst = cv2.cornerHarris(img, 2, 3, 0.04)
            img[dst > 0.01 * dst.max()] = 255  # Changed this line
            cv2.imwrite(output_path, img)

        return redirect(url_for('processed', filename=filename))

    return render_template('process.html', filename=filename)


@app.route('/processed/<filename>')
def processed(filename):
    # Construct the relative file path
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    # Print the list of files in the processed folder
    print("Files in processed folder: ", os.listdir(app.config['PROCESSED_FOLDER']))

    # Print the relative file path
    print("Relative file path: ", file_path)

    # Check if the file exists
    if not os.path.isfile(file_path):
        return "File not found. Please wait a moment and refresh the page."

    # Convert the relative path to an absolute path
    absolute_file_path = os.path.abspath(file_path)

    # Send the file
    return send_file(absolute_file_path)

if __name__ == '__main__':
    app.run(debug=True)