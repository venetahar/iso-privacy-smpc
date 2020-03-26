from mnist.common.mnist_data_loader import MnistDataLoader


class PrivateMnistDataLoader(MnistDataLoader):

    def __init__(self, party_one, party_two, crypto_provider):
        super(PrivateMnistDataLoader, self).__init__()

        self.private_test_loader = []

        for data, labels in self.test_loader:
            self.private_test_loader.append((
                data.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider),
                labels.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider)
            ))
