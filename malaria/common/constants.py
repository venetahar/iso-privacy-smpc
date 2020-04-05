MALARIA_NORM_MEAN = [0.5, 0.5, 0.5]
MALARIA_NORM_STD = [0.5, 0.5, 0.5]
IMG_RESIZE = (32, 32)

TRAIN_PERCENTAGE = 0.8
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 8

TRAINING_PARAMS = {
    'learning_rate': 0.001,
    'momentum': 0.9,
    'num_epochs': 10
}

CONVNET_MODEL_PATH = '../models/alice_conv_model.pth'
