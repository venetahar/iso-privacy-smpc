MNIST_DIMENSIONS = (28, 28, 1)
NUM_CLASSES = 10

BATCH_SIZE = 128
TEST_BATCH_SIZE = 200

TRAINING_PARAMS = {
        'learning_rate': 0.001,
        'momentum': 0.9,
        'num_epochs': 15,
        'optim': 'Adam',
        'test_batch_size': TEST_BATCH_SIZE
}
