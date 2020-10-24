from flask import Flask, render_template, request
from model import detect_mask_image as detect

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/api')
def api():
    image = request.files['image']
    is_masked = detect.detect_and_predict_mask(image)


if __name__ == '__main__':
    app.run()
