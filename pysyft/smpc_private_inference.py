from common.mnist_training import MnistTraining
from pysyft.smpc_mnist import SMPCMnist


def train_plain_model():
    mnist_training = MnistTraining()
    mnist_training.train()
    mnist_training.save_model('../models/alice_model.pth')


# train_plain_model()

smpc_mnist = SMPCMnist()
smpc_mnist.encrypt_evaluate_model('../models/alice_model.pth')
