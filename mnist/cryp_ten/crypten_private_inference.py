from mnist.common.mnist_training import MnistTraining
from mnist.cryp_ten.crypten_mnist import CrypTenMnist

LENET_MODEL_PATH = '../models/alice_model.pth'
CONVNET_MODEL_PATH = '../models/alice_conv_model.pth'


def train_plain_model(model_type='LeNet', model_path=LENET_MODEL_PATH):
    mnist_training = MnistTraining(model_type)
    mnist_training.train()
    mnist_training.save_model(model_path)
    mnist_training.save_labels('./data/bob')


def evaluaute_encrypted(model_type='LeNet', model_path=LENET_MODEL_PATH):
    crypten_model = CrypTenMnist(model_type)
    crypten_model.encrypt_evaluate_model(model_path, './data/bob_test.pth',
                                         './data/bob_test_labels.pth')


# train_plain_model()
# evaluaute_encrypted()

# train_plain_model('ConvNet', CONVNET_MODEL_PATH)
evaluaute_encrypted('ConvNet', CONVNET_MODEL_PATH)
