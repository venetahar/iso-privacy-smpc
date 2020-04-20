from mnist.common.mnist_data_loader import MnistDataLoader


class PrivatePySyftMnistDataLoader(MnistDataLoader):
    """
    A Private MNIST Data loader responsible for encrypting and sharing the data.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates a PrivateMnistDataLoader.
        """
        super(PrivatePySyftMnistDataLoader, self).__init__(data_path, test_batch_size)
        self.private_test_loader = []

    def encrypt_test_data(self, party_one, party_two, crypto_provider):
        """
        Encrypts the test data.
        :param party_one: Party one.
        :param party_two: Party two.
        :param crypto_provider: The crypto provider.
        """
        for data, labels in self.test_loader:
            self.private_test_loader.append((
                data.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider),
                labels.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider)
            ))
