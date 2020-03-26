from common.mnist_training import MnistTraining
from pysyft.pysyft_mnist import PysyftMnist


def train_plain_model():
    mnist_training = MnistTraining()
    mnist_training.train()
    mnist_training.save_model('../models/alice_model.pth')


# train_plain_model()

smpc_mnist = PysyftMnist()
# smpc_mnist.encrypt_evaluate_model('../models/alice_model.pth')
smpc_mnist.encrypt_evaluate_model('../models/alice_conv_model.pth')
