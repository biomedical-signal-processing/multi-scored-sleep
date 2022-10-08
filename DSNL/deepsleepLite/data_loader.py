import os
import numpy as np

class DataLoader(object):
  
    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file, cond_prob): 
        """Load data and labels from a npz file."""
        cdir = os.getcwd()
        os.chdir(self.data_dir)
        with np.load(npz_file, allow_pickle=True) as f:
            data = f["x"] # EEG channel  
            labels = f["y"] # labels
            sampling_rate = int(f["fs"]) # sampling rate
            if cond_prob:
                labels_smoothed = f["y_smoothed"] # LSSC distribution
        os.chdir(cdir)
        return data, labels, labels_smoothed, sampling_rate

    def load_testdata_cv(self, data_dir):
        """Load test data and labels in predict.py from test list."""
        f = np.load(data_dir)
        test_files = f["test_files"].tolist()
        data = []
        labels = []
        labels_smoothed = []
        for file in test_files:
            x, y, y_smoothed, sampling_rate = self._load_npz_file(file,cond_prob=True)

            # Reshape the data to match the input of the model - conv2d
            x = np.squeeze(x)
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            data.append(x)
            labels.append(y)
            labels_smoothed.append(y_smoothed)

        return data, labels, labels_smoothed, test_files
