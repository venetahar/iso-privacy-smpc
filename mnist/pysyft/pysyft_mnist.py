from mnist.common.constants import CONVNET_MODEL_PATH, TEST_BATCH_SIZE
from mnist.common.mnist_training import train_mnist_model
from common.pysyft.pysyft_private_inference import PysyftPrivateInference
from mnist.pysyft.private_mnist_data_loader import PrivateMnistDataLoader

should_train = True
if should_train:
    train_mnist_model('ConvNet', CONVNET_MODEL_PATH)

smpc_mnist = PysyftPrivateInference(PrivateMnistDataLoader(), parameters={'test_batch_size': TEST_BATCH_SIZE})
smpc_mnist.encrypt_evaluate_model(CONVNET_MODEL_PATH)
