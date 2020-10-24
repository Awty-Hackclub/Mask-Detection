from flask import Flask, render_template, request
from model import detect_mask_image as detect


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
