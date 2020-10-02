import torch
from flask import Flask, request, abort, Response, json
from werkzeug.utils import secure_filename

import sys
sys.path.append("../..")

# required to successfully load saved model
from malaria.common.conv_pool_model import ConvPoolModel
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel

# preprocess input images
from malaria.common.malaria_data_loader import malaria_transform
from mnist.common.mnist_data_loader import mnist_transform
from common.utils.pytorch_utils import *

app = Flask(__name__)

WORKDIR = 'rest/workdir'

models = {
    "mnist_conv_model": "mnist/models/alice_conv1_model.pth",
    "mnist_fc_model": "mnist/models/alice_fc3_model.pth",
    "malaria_conv_model": "malaria/models/alice_convpool_model.pth"
}

transforms = {
    "mnist": mnist_transform,
    "malaria": malaria_transform
}


def save_file_received(file_obj):
    filename = secure_filename(file_obj.filename)
    file_obj.save(filename)


def extract_model_key():
    if 'model' in request.form:
        model_key = request.form["model"]
    else:
        print('No model part in request!')
        abort(500)

    if model_key not in models:
        print('Unknown model requested!')
        abort(500)
    return model_key


def load_model(model_filename):
    from common.utils.sys_utils import cd

    with cd(WORKDIR):
        return torch.load(model_filename)


def load_new_model(model_key):

    global model, selected_model

    if selected_model != model_key:
        model_filename = models[model_key]
        model = load_model(model_filename)
        selected_model = model_key
    return model, selected_model


def load_image(image_file, model_type):
    from PIL import Image

    img = Image.open(image_file)
    transform = transforms[model_type]()

    timg = transform(img)
    timg.unsqueeze_(0)

    return timg


def test(in_model, data):
    in_model.eval()
    with torch.no_grad():
        outputs = in_model(data)
        prediction = outputs.argmax(dim=1)

        return prediction


def run_inference(input_data, model_key):
    global model, selected_model

    model, selected_model = load_new_model(model_key)
    prediction = test(model, input_data)
    return prediction


@app.route('/api/predict', methods=['POST'])
def predict_single():
    global model, selected_model

    # handle input data file
    if 'input_file' not in request.files:
        print('No file part in request!')
        abort(500)
    file = request.files['input_file']

    if file.filename == '':
        print("No file uploaded!")
        abort(500)

    save_file_received(file_obj=file)

    # handle model selection parameter
    model_key = extract_model_key()

    model_type = model_key.split('_')[0]
    input_image = load_image(file, model_type)

    prediction = run_inference(input_image, model_key)

    response = Response(response=json.dumps(torch2json(prediction)),
                        status=200,
                        mimetype='application/json')

    return response


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():

    input_batch_json = request.form["batch"]
    input_batch = json2torch(input_batch_json)

    model_key = extract_model_key()

    prediction = run_inference(input_batch, model_key)

    response = Response(response=torch2json(prediction),
                        status=200,
                        mimetype='application/json')

    return response


# Place server initialisation code here
selected_model = "mnist_fc_model"
model = load_model(models[selected_model])


if __name__ == '__main__':
    app.run()
