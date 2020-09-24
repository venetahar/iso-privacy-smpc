from malaria.common.malaria_data_loader import MalariaDataLoader


class PrivateMalariaDataLoader(MalariaDataLoader):
    """
    A private malaria data loader responsible for sharing the data with the respective parties.
    """

    def __init__(self, data_path, test_batch_size):
        """
        Creates a PrivateMalariaDataLoader.
        :param data_path: The data path where the files are located.
        """
        super(PrivateMalariaDataLoader, self).__init__(data_path, test_batch_size)
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
