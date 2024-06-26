# app.py
from flask import Flask, flash, render_template, request, redirect, url_for, send_from_directory,send_file
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
import os
import cv2
import numpy as np
import secrets
import time
from .Canny import canny_edge_detector
from .Hough import hough_line_transform
from .Harris import harris_corner_detector
from .Hough_ellipse import randomized_hough_ellipse_transform

class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileAllowed(['jpg', 'png', 'jpeg', 'webp'], 'Only .jpg, .png, .jpeg, .webp formats are allowed!')])

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
            if file:
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.makedirs(os.path.dirname(upload_path), exist_ok=True)
                file.save(upload_path)
                return redirect(url_for('process', filename=filename))
            else:
                flash('Please select an image first.', 'error')
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
            canny_low_threshold = int(request.form.get('canny_low_threshold'))
            canny_high_threshold = int(request.form.get('canny_high_threshold'))
            canny_sigma = float(request.form.get('canny_sigma'))
            canny_ksize = int(request.form.get('canny-kernel-size', '3'))  
            edges = canny_edge_detector(img, canny_sigma, canny_ksize, canny_low_threshold, canny_high_threshold)
            cv2.imwrite(output_path, edges)
        elif operation == 'hough':
            theta_resolution = int(request.form.get('theta_resolution'))
            Hough_threshold = int(request.form.get('num_peaks'))
            hough_low_threshold = int(request.form.get('hough_low_threshold'))
            hough_high_threshold = int(request.form.get('hough_high_threshold'))
            hough_ksize = int(request.form.get('hough-kernel-size', '3')) 
            hough_sigma = float(request.form.get('hough_sigma'))
            # hough_line_img = hough_line_transform(img, (np.pi/180) * theta_resolution, Hough_threshold, hough_low_threshold, hough_high_threshold, hough_ksize, hough_sigma)
            hough_line_img = hough_line_transform(img, (np.pi/180) * theta_resolution, Hough_threshold, hough_low_threshold, hough_high_threshold, hough_ksize, hough_sigma)
            cv2.imwrite(output_path, hough_line_img)  # Save color image
        elif operation == 'harris':
            harris_threshold = float(request.form.get('harris_threshold', '0.01'))

            # Convert image to float32 for processing
            img_float = np.float32(img)
            corners = harris_corner_detector(img_float, 3, 0.04, harris_threshold)
            
            # Convert grayscale image back to color
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Visualize corners
            for y, x in zip(*corners):
                cv2.circle(img_color, (x, y), 2, (0,0,255), 1)
            
            cv2.imwrite(output_path, img_color)
        elif operation == 'hough-ellipse':
            print("Entered 'hough-ellipse' operation")  # Debugging line
            sigma = float(request.form.get('hough_ellipse_sigma', 4))  # default value is 2.0
            low_threshold = float(request.form.get('hough_ellipse_low_threshold', 20))  # default value is 0.55
            high_threshold = float(request.form.get('hough_ellipse_high_threshold', 50))  # default value is 0.8
            # Assuming you have a function hough_ellipse_transform for ellipse detection
            print("Calling hough_ellipse_transform...")  # Debugging line
            ellipse_img = randomized_hough_ellipse_transform(img, canny_t1= low_threshold,
            canny_t2= high_threshold, canny_sigma= sigma)
            print("hough_ellipse_transform completed")  # Debugging line
            p, q, a, b, angle, score = ellipse_img
            img_copy = img.copy()
            final_img = cv2.ellipse(img_copy, (int(p), int(q)), (int(a), int(b)), angle * 180 / np.pi, 0, 360, color=(0, 255, 0), thickness=2)
            cv2.imwrite(output_path, final_img)
            print(f"Image written to {output_path}")  # Debugging line
        return redirect(url_for('processed', filename=filename))

    return render_template('process.html', filename=filename)

@app.route('/processed/<filename>')
def processed(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    print("Files in processed folder: ", os.listdir(app.config['PROCESSED_FOLDER']))

    print("Relative file path: ", file_path)

    if not os.path.isfile(file_path):
        return "File not found. Please wait a moment and refresh the page."

    absolute_file_path = os.path.abspath(file_path)
    return send_file(absolute_file_path)

if __name__ == '__main__':
    app.run(debug=True)