# Definition of useful variables:
#Labels values:
NUM_CLASSES = 5
EPOCH_SEC_LEN = 30  # seconds
SEQ_OF_EPOCHS = 3  # number of epochs in a sequence / length L
SAMPLING_RATE = 100.0 # fs del dataset IS-RC

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


def print_n_samples_each_class(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))
