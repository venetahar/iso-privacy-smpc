import os
from shutil import move

import numpy as np

np.random.seed(0)

PARASITIZED = 'Parasitized'
UNINFECTED = 'Uninfected'


class MalariaDatasetUtils:

    @staticmethod
    def split_dataset(data_path, split_size=0.9, training_folder='training', testing_folder='testing'):

        training_parasitized_path = os.path.join(data_path, training_folder, PARASITIZED)
        training_uninfected_path = os.path.join(data_path, training_folder, UNINFECTED)
        testing_parasitized_path = os.path.join(data_path, testing_folder, PARASITIZED)
        testing_uninfected_path = os.path.join(data_path, testing_folder, UNINFECTED)

        MalariaDatasetUtils.create_directory(training_parasitized_path)
        MalariaDatasetUtils.create_directory(training_uninfected_path)
        MalariaDatasetUtils.create_directory(testing_parasitized_path)
        MalariaDatasetUtils.create_directory(testing_uninfected_path)

        parasitized_file_path = os.path.join(data_path, PARASITIZED)
        parasitized_files = MalariaDatasetUtils.get_files_in_directory(parasitized_file_path)

        uninfected_file_path = os.path.join(data_path, UNINFECTED)
        uninfected_files = MalariaDatasetUtils.get_files_in_directory(uninfected_file_path)
        MalariaDatasetUtils.create_dataset_split_for_files(parasitized_files, parasitized_file_path,
                                                           split_size=split_size,
                                                           training_folder=training_parasitized_path,
                                                           testing_folder=testing_parasitized_path)
        MalariaDatasetUtils.create_dataset_split_for_files(uninfected_files, uninfected_file_path,
                                                           split_size=split_size,
                                                           training_folder=training_uninfected_path,
                                                           testing_folder=testing_uninfected_path)

    @staticmethod
    def create_dataset_split_for_files(files, data_path, training_folder='training', testing_folder='testing',
                                       split_size=0.9):
        num_samples = len(files)
        indices = list(range(num_samples))
        num_train_samples = int(np.floor(split_size * num_samples))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[:num_train_samples], indices[num_train_samples:]

        training_set = files[train_indices]
        testing_set = files[test_indices]

        MalariaDatasetUtils.move_files(data_path, training_folder, training_set)

        MalariaDatasetUtils.move_files(data_path, testing_folder, testing_set)

    @staticmethod
    def move_files(data_path, target_path, dataset_files):
        for filename in dataset_files:
            this_file = os.path.join(data_path, filename)
            destination = os.path.join(target_path, filename)
            move(this_file, destination)

    @staticmethod
    def get_files_in_directory(directory):
        files = []
        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            if os.path.getsize(file) > 0:
                files.append(filename)
            else:
                print(filename + " is zero length, so ignoring.")
        return np.array(files)

    @staticmethod
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)


MalariaDatasetUtils.split_dataset('data/cell_images')
