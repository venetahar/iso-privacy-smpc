from malaria.common.conv_pool_model import ConvPoolModel
from mnist.common.conv_model import ConvModel
from mnist.common.fully_connected_model import FullyConnectedModel


class ModelFactory:
    """
    A Model Factory.
    """

    @staticmethod
    def create_model(model_type, input_shape, num_classes):
        """
        Returns a model of the appropriate type.
        :param model_type: The model type.
        :param input_shape: The input shape.
        :param num_classes: The number of classes.
        :return: An instantiated model.
        """
        if model_type == 'FullyConnected3':
            model = FullyConnectedModel(input_shape, hidden_units=[128, 128], num_classes=num_classes)
        elif model_type == 'Conv1':
            model = ConvModel(input_shape, out_channels=5, kernel_size=5, stride=2, padding=0, avg_pool_size=2,
                              linear_units=100, num_classes=10)
        elif model_type == 'Conv2Pool2':
            model = ConvPoolModel(input_shape=input_shape, num_classes=2, conv_kernel_sizes=[5, 5], channels=[36, 36],
                                  avg_pool_sizes=[2, 2], fc_units=[72])
        else:
            raise ValueError("Invalid model_type provided. ")

        print(model)
        return model
