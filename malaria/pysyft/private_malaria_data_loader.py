from malaria.common.malaria_data_loader import MalariaDataLoader


class PrivateMalariaDataLoader(MalariaDataLoader):

    def __init__(self, data_path, should_load_split=True):
        super(PrivateMalariaDataLoader, self).__init__(data_path, should_load_split)
        self.private_test_loader = []

    def encrypt_data(self, party_one, party_two, crypto_provider):
        for data, labels in self.test_loader:
            self.private_test_loader.append((
                data.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider),
                labels.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider)
            ))
