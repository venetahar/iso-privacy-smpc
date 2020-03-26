from torch import optim, save
from torch.nn import CrossEntropyLoss

from mnist.common.constants import LEARNING_RATE, MOMENTUM, NUM_EPOCHS
from mnist.common.conv_plain_text_model import ConvPlainTextNet
from mnist.common.lenet_plain_text_model import PlainTextNet
from mnist.common.mnist_data_loader import MnistDataLoader


class MnistTraining:

    def __init__(self, model_type='LeNet'):
        if model_type == 'ConvNet':
            self.model = ConvPlainTextNet()
        else:
            self.model = PlainTextNet()
        self.mnist_data_loader = MnistDataLoader()
        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    def train(self):
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for index, data in enumerate(self.mnist_data_loader.train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()
            print('[%d] loss: %.3f' % (epoch + 1,  running_loss / len(self.mnist_data_loader.train_loader)))

    def save_labels(self, data_path):
        save(self.mnist_data_loader.test_set.data.unsqueeze(1), data_path + "_test.pth")
        save(self.mnist_data_loader.test_set.targets, data_path + "_test_labels.pth")

    def save_model(self, model_path):
        save(self.model, model_path)
