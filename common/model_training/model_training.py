from torch import optim, save
from torch.nn import CrossEntropyLoss


class ModelTraining:
    """
    Class for model training.
    """

    def __init__(self, model, data_loader, training_parameters, criterion=CrossEntropyLoss()):
        """
        Initialises a ModelTraining object. 
        :param model: The model to train.
        :param data_loader: The data loader.
        :param training_parameters: The training parameters. 
        """
        self.model = model
        self.data_loader = data_loader
        self.training_parameters = training_parameters
        self.criterion = criterion
        if self.training_parameters['optim'] == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.training_parameters['learning_rate'])
            print("Using Adam")
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.training_parameters['learning_rate'],
                                       momentum=self.training_parameters['momentum'])

    def train(self):
        """
        Trains a model and prints the loss.
        """
        for epoch in range(self.training_parameters['num_epochs']):
            running_loss = 0.0
            for index, data in enumerate(self.data_loader.train_loader):
                inputs, labels = data

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                running_loss += loss.item()

                self.optimizer.step()
            print('[%d] loss: %.3f' % (epoch + 1,  running_loss / len(self.data_loader.train_loader)))

    def evaluate_plain_text(self):
        """
        Evaluates the plain text accuracy of a model.
        """
        correct_predictions = 0
        self.model.eval()
        total_predictions = 0
        for index, (inputs, labels) in enumerate(self.data_loader.test_loader):
            outputs = self.model(inputs)
            pred = outputs.argmax(dim=1)
            correct_predictions += pred.eq(labels.view_as(pred)).sum()
            total_predictions += len(labels)

        accuracy = 100.0 * correct_predictions / total_predictions
        print('Plaintext test set: Accuracy: {}/{} ({:.4f}%)'.format(correct_predictions, total_predictions, accuracy))

    def save_model(self, model_path):
        """
        Saves the model to a specified path.
        :param model_path: Model path for the saved model.
        """
        save(self.model, model_path)
