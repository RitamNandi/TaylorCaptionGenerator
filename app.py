#flask --app app run

from flask import Flask, render_template, request, Response
from file import process_image
import os

app = Flask(__name__, template_folder="templates", static_folder="uploads", static_url_path="/uploads")

app.config['STATIC_FOLDER'] = 'uploads'


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Call the image processing function from file.py

        closest_lyric, album_name = process_image(filename)
        uploaded_image_url = file.filename
        return render_template('result.html', closest_lyric=closest_lyric, uploaded_image=uploaded_image_url, album_name=album_name)


if __name__ == '__main__':
    app.run(debug=True)
