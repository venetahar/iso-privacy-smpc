from eyepacs.common.eyepacs_data_loader import EyepacsDataLoader


class PrivateEyepacsDataLoader(EyepacsDataLoader):

    def __init__(self, party_one, party_two, crypto_provider, train_csv_path, test_csv_path):
        super(PrivateEyepacsDataLoader, self).__init__(train_csv_path, test_csv_path)

        self.private_test_loader = []

        for data, labels in self.test_loader:
            self.private_test_loader.append((
                data.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider),
                labels.fix_prec().share(party_one, party_two, crypto_provider=crypto_provider)
            ))
