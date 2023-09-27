import os
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template

MODEL_PATH = 'Deployment-Deep-Learning-Model/saved_model/model_vgg16.h5'

model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img = img.reshape(1, 256, 256, 3)
    predictions = model.predict(img)[0][0]
    
    return predictions

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)

        if preds < 0.5:
            result = "Bad"

        else:
            result = "Good"            
        return result
    
    return None

if __name__ == '__main__':
    app.run(debug=True)

