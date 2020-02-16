from cryp_ten.cryp_ten_mnist import CrypTenMnist
from common.mnist_training import MnistTraining


def train_plain_model():
    mnist_training = MnistTraining()
    mnist_training.train()
    mnist_training.save_model('../models/alice_model.pth')
    mnist_training.save_labels('./data/bob')


def evaluaute_encrypted():
    crypten_model = CrypTenMnist()
    crypten_model.encrypt_evaluate_model('../models/alice_model.pth', './data/bob_test.pth',
                                         './data/bob_test_labels.pth')


# train_plain_model()
evaluaute_encrypted()
