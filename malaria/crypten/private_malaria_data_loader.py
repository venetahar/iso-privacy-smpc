from crypten import cryptensor

from malaria.common.malaria_data_loader import MalariaDataLoader


class PrivateMalariaDataLoader(MalariaDataLoader):
    """
    A private malaria data loader responsible for sharing the data with the respective parties.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates a PrivateMalariaDataLoader.
        :param data_path: The data path where the files are located..
        """
        super(PrivateMalariaDataLoader, self).__init__(data_path, test_batch_size)
        self.private_test_loader = []

    def encrypt_test_data(self, data_owner):
        """
        Encrypts the test data.
        :param data_owner: The data owner
        """
        for data, labels in self.test_loader:
            self.private_test_loader.append((
                cryptensor(data, src=data_owner),
                cryptensor(labels, src=data_owner)
            ))
