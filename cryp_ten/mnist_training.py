import os

from torch import optim, save
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from cryp_ten.constants import NUM_EPOCHS, LEARNING_RATE, MOMENTUM
from cryp_ten.plain_text_model import PlainTextNet
from torchvision import datasets


class MnistTraining:

    def __init__(self):
        self.model = PlainTextNet()
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

        self.test_set = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.test_set, batch_size=4, shuffle=False, num_workers=2)

        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    def train(self):
        for epoch in range(NUM_EPOCHS):
            running_loss = 0.0
            for index, data in enumerate(self.train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()
            print('[%d] loss: %.3f' % (epoch + 1,  running_loss / len(self.train_loader)))

    def save_labels(self, data_path):
        save(self.test_set.data, data_path + "_test.pth")
        save(self.test_set.targets, data_path + "_test_labels.pth")

    def save_model(self, model_path):
        save(self.model, model_path)
