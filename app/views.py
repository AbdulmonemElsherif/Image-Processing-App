# views.py
from flask import render_template, request, redirect, url_for
from app import app
from .forms import UploadForm
from .models import process_image, canny_edge_detection, hough_line_detection, harris_corner_detection

@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        image_path = form.image.data
        process_image(image_path)
    return render_template('upload.html', form=form)

@app.route('/process/<filename>', methods=['GET', 'POST'])
def process(filename):
    if request.method == 'POST':
        operation = request.form.get('operation')
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        if operation == 'canny':
            kernel_size = int(request.form.get('canny_kernel_size', 3))
            low_threshold = int(request.form.get('canny_low_threshold', 100))
            high_threshold = int(request.form.get('canny_high_threshold', 200))
            canny_edge_detection(input_path, output_path, kernel_size, low_threshold, high_threshold)
        elif operation == 'hough':
            resolution = int(request.form.get('hough_resolution', 1))
            num_lines = int(request.form.get('hough_lines', 10))
            hough_line_detection(input_path, output_path, resolution, num_lines)
        elif operation == 'harris':
            threshold = float(request.form.get('harris_threshold', 0.04))
            harris_corner_detection(input_path, output_path, threshold)

        return redirect(url_for('processed', filename=filename))

    return render_template('process.html', filename=filename)

# ...rest of the code...
