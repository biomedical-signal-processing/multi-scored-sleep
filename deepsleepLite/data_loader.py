import os
import numpy as np
from deepsleepLite.utils import get_sequences, get_sequences_prob
import random




class DataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file_DODO(self, npz_file, cond_prob):
        """Load data and labels from a npz file."""
        # print(f"{self.data_dir}/{npz_file}")
        f = np.load(f"{self.data_dir}/{npz_file}", allow_pickle=True)
        data = f["x"]
        labels = f["y"]
        if cond_prob:
            labels_smoothed = f["y_smoothed"]
        sampling_rate = int(f["fs"])
        return data, labels, labels_smoothed, sampling_rate

    def _load_npz_list_files_DODO(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        labels_smoothed = []
        sampling_rate = 128  # fs del database IS-RC
        for npz_f in npz_files:
            print(f"Loading {npz_f} ...")
            tmp_data, tmp_labels, tmp_labels_smoothed, sampling_rate = self._load_npz_file_DODO(npz_f)

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)
            tmp_labels_smoothed = tmp_labels_smoothed.astype(np.float32)

            data.append(tmp_data)
            labels.append(tmp_labels)
            labels_smoothed.append(tmp_labels_smoothed)

        return data, labels, labels_smoothed, sampling_rate

    def load_files_cv(self, dodh=False, dodo=False, dodo_dodh=False):

        if dodh:
            n_valid = 6
        elif dodo:
            n_valid = 12
            test = ['Opsg_0_.npz', 'Opsg_7_.npz', 'Opsg_13_.npz', 'Opsg_20_.npz', 'Opsg_27_.npz']
        elif dodo_dodh:
            n_valid = 16

        allfiles = os.listdir(self.data_dir)
        allfiles.sort(key=lambda x: int(x.split("_")[1]))
        # Database
        if self.n_folds > len(allfiles):
            raise Exception(" k-fold > n_subject!!")
        else:
            n_test = round(len(allfiles) / self.n_folds)  # number of subject in validation
            test_idx_end = (self.fold_idx + 1) * n_test
            test_idx_start = self.fold_idx * n_test

            # Splitting files:
            test_files = allfiles[test_idx_start:test_idx_end]
            train_files = list(set(allfiles) - set(test_files))
            valid_files = random.sample(train_files, n_valid)  # 20% del train è scelto come valid
            train_files = list(set(train_files) - set(valid_files))

            if dodo and self.fold_idx <=4:
                test_files = list(set(test_files) - set([test[self.fold_idx]]))
            if dodo and self.fold_idx == 9:
                test_files.extend(test)
                train_files = list(set(allfiles) - set(test_files))
                valid_files = random.sample(train_files, n_valid)  # 20% del train è scelto come valid
                train_files = list(set(train_files) - set(valid_files))

            # Sorting:
            test_files.sort(key=lambda x: int(x.split("_")[1]))
            valid_files.sort(key=lambda x: int(x.split("_")[1]))
            train_files.sort(key=lambda x: int(x.split("_")[1]))

        return train_files, valid_files, test_files


    def _load_npz_file_ISRC(self, npz_file, cond_prob): # cond_prob = empirical distribution
        """Load data and labels from a npz file."""
        cdir = os.getcwd()
        os.chdir(self.data_dir)
        with np.load(npz_file, allow_pickle=True) as f:
            data = f["x"]  # single channel EEG 'C4-A1'
            labels = f["y"]
            sampling_rate = int(f["fs"])
            if cond_prob:
                labels_smoothed = f["y_smoothed"]
        os.chdir(cdir)
        return data, labels, labels_smoothed, sampling_rate

    def _load_npz_list_files(self, npz_files, cond_prob):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        labels_smoothed = []
        fs = None
        for npz_f in npz_files:
            print(f"Loading {npz_f} ...")
            # TODO condition on database to be loaded
            tmp_data, tmp_labels, tmp_labels_smoothed, sampling_rate = self._load_npz_file_DODO(npz_f, cond_prob)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)
            tmp_labels_smoothed = tmp_labels_smoothed.astype(np.float32)

            data.append(tmp_data)
            labels.append(tmp_labels)
            labels_smoothed.append(tmp_labels_smoothed)

        return data, labels, labels_smoothed, sampling_rate

    def load_data_sequences(self, input_files, seq_length, train, cond_prob):

        subject_files = input_files
        subject_files.sort(key=lambda x: int(x.split("_")[1]))

        # Load training set
        if train is True:
            # Load training set
            print(f"\n========== [Fold-{self.fold_idx}] ==========\n")
            print("Load training set:")
        else:
            # Load Validation set
            print(f"\n========== [Fold-{self.fold_idx}] ==========\n")
            print("Load validation set:")

        data, label, labels_smoothed, sampling_rate = self._load_npz_list_files(npz_files=subject_files,
                                                                                cond_prob=cond_prob)
        # Extract sequences of length L=seq_length
        data, label = get_sequences(
            x=data, y=label, seq_length=seq_length
        )
        if cond_prob is True:
            labels_smoothed = get_sequences_prob(
                y=labels_smoothed, seq_length=seq_length
            )
            labels_smoothed = np.vstack(labels_smoothed)
        if train is True:
            print("Training set: n_psg={}".format(len(data)))
        else:
            print("Validation set: n_psg={}".format(len(data)))

        data = np.vstack(
            data)  # unisco tutte le epoche per righe, prima di questa operazione tutti i pazienti erano suddivisi ognuno con le sue epoche
        label = np.vstack(label)

        print(f"Number of examples = {len(data)}")

        print(" ")

        return data, label, sampling_rate, labels_smoothed

    def load_ISRC_testdata_cv(self, data_dir):
        """Load test data and labels in prediction.py from test list."""

        f = np.load(data_dir)
        test_files = f["test_files"].tolist()
        data = []
        labels = []
        labels_smoothed = []
        for file in test_files:
            x, y, y_smoothed, sampling_rate = self._load_npz_file_ISRC(file,cond_prob=True)

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            data.append(x)
            labels.append(y)
            labels_smoothed.append(y_smoothed)

        return data, labels, labels_smoothed, test_files

