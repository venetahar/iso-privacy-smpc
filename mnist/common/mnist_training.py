from common.model_training.model_training import ModelTraining
from common.utils.data_utils import DataUtils
from mnist.common.constants import LENET_MODEL_PATH, TRAINING_PARAMS
from mnist.common.conv_plain_text_model import ConvPlainTextNet
from mnist.common.lenet_plain_text_model import PlainTextNet
from mnist.common.mnist_data_loader import MnistDataLoader


def train_mnist_model(model_type='LeNet', model_path=LENET_MODEL_PATH):
    mnist_data_loader = MnistDataLoader()
    if model_type == 'ConvNet':
        model = ConvPlainTextNet()
    else:
        model = PlainTextNet()
    mnist_training = ModelTraining(model, mnist_data_loader, training_parameters=TRAINING_PARAMS)
    mnist_training.train()
    mnist_training.save_model(model_path)

    DataUtils.save_labels('./data/bob', mnist_data_loader.test_set)
