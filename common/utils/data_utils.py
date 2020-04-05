from torch import save


class DataUtils:
    """
    Common class for data utilities.
    """

    @staticmethod
    def save_labels(data_path, data_set):
        save(data_set.data.unsqueeze(1), data_path + "_test.pth")
        save(data_set.targets, data_path + "_test_labels.pth")
