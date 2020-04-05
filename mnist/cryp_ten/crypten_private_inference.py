from mnist.common.mnist_training import train_mnist_model
from mnist.cryp_ten.crypten_mnist import CrypTenMnist
from mnist.common.constants import LENET_MODEL_PATH, CONVNET_MODEL_PATH


def evaluate_encrypted(model_type='LeNet', model_path=LENET_MODEL_PATH):
    crypten_model = CrypTenMnist(model_type)
    crypten_model.encrypt_evaluate_model(model_path, './data/bob_test.pth',
                                         './data/bob_test_labels.pth')


should_train = False
if should_train:
    train_mnist_model('ConvNet', CONVNET_MODEL_PATH)
evaluate_encrypted('ConvNet', CONVNET_MODEL_PATH)
