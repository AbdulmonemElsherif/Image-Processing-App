# views.py
from flask import render_template, request
from app import app
from .forms import UploadForm
from .models import process_image

@app.route('/', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        image_path = form.image.data
        process_image(image_path)
    return render_template('upload.html', form=form)