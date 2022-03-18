
    import os
    import matplotlib.pyplot as plt
    import numpy as np


    def compute_calibration(true_labels, pred_labels, confidences, num_bins):
        """Collects predictions into bins used to draw a reliability diagram.
        Arguments:
            true_labels: the true labels for the test examples
            pred_labels: the predicted labels for the test examples
            confidences: the predicted confidences for the test examples
            num_bins: number of bins
        The true_labels, pred_labels, confidences arguments must be NumPy arrays;
        pred_labels and true_labels may contain numeric or string labels.
        For a multi-class model, the predicted label and confidence should be those
        of the highest scoring class.
        Returns a dictionary containing the following NumPy arrays:
            accuracies: the average accuracy for each bin
            confidences: the average confidence for each bin
            counts: the number of examples in each bin
            bins: the confidence thresholds for each bin
            avg_accuracy: the accuracy over the entire test set
            avg_confidence: the average confidence over the entire test set
            expected_calibration_error: a weighted average of all calibration gaps
            max_calibration_error: the largest calibration gap across all bins
        """
        assert (len(confidences) == len(pred_labels))
        assert (len(confidences) == len(true_labels))
        assert (num_bins > 0)

        bin_size = 1.0 / num_bins
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        indices = np.digitize(confidences, bins, right=True)

        bin_accuracies = np.zeros(num_bins, dtype=np.float)
        bin_confidences = np.zeros(num_bins, dtype=np.float)
        bin_counts = np.zeros(num_bins, dtype=np.int)

        for b in range(num_bins):
            selected = np.where(indices == b + 1)[0]
            if len(selected) > 0:
                bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
                bin_confidences[b] = np.mean(confidences[selected])
                bin_counts[b] = len(selected)

        avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
        avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
        mce = np.max(gaps)

        return {"accuracies": bin_accuracies,
                "confidences": bin_confidences,
                "counts": bin_counts,
                "bins": bins,
                "avg_accuracy": avg_acc,
                "avg_confidence": avg_conf,
                "expected_calibration_error": ece,
                "max_calibration_error": mce}


    def read_results(path):
        f = np.load(f"{path}/performance_overall.npz")
        y_true = f["y_true"]
        y = f["y_pred"]
        for fold_idx in range(10):
            f = np.load(f"{path}/output_fold{fold_idx}.npz", allow_pickle=True)
            p = f["prob_pred"]
            for i in range(len(p)):
                if i == 0:
                    pp = np.array(p[i])
                else:
                    pp = np.vstack((pp, np.array(p[i])))
            if fold_idx == 0:
                prob = pp
            else:
                prob = np.vstack((prob, pp))
        y_conf = np.max(prob, axis=1)

        return y_true, y, y_conf, prob


    def all_models_calib(base_path, n_bins):
        ece = []
        conf = []
        acc = []
        folders = os.listdir(base_path)
        for fold in folders:
            try:
                y_true, y, y_conf, prob = read_results(f"{base_path}/{fold}")
                diz = compute_calibration(y_true, y, y_conf, n_bins)
                ece.append(diz["expected_calibration_error"])
                conf.append(diz["avg_confidence"])
                acc.append(diz["avg_accuracy"])
            except:
                pass

        return ece, conf, acc


    if __name__ == "__main__":
        base_path_LSE = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/LSE"
        base_path_LSU = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/LSU"
        base_path_base = "/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/prediction/base"
        n_bins = 20
        LSE_ece, LSE_conf, LSE_acc = all_models_calib(base_path_LSE, n_bins)
        LSU_ece, LSU_conf, LSU_acc = all_models_calib(base_path_LSU, n_bins)
        base_ece, base_conf, base_acc = all_models_calib(base_path_base, n_bins)
        print(f"LSE:\n ECE:{np.round(LSE_ece, 3)}\n ACC:{np.round(LSE_acc, 3)}\n CONF:{np.round(LSE_conf, 3)}\n")
        print(f"LSU:\n ECE:{np.round(LSU_ece, 3)}\n ACC:{np.round(LSU_acc, 3)}\n CONF:{np.round(LSU_conf, 3)}\n")
        print(f"base:\n ECE:{np.round(base_ece, 3)}\n ACC:{np.round(base_acc, 3)}\n CONF:{np.round(base_conf, 3)}\n")

        plt.plot(LSU_ece, "-^", markevery=[3], color="dimgrey", label="LSU")
        # plt.plot(DSL_two_ece[1:],"-^",markevery=[5],color="navy")
        plt.axhline(y=base_ece[0], color='deepskyblue', linestyle='-', linewidth="1", label="base")
        plt.plot(LSE_ece, "-^", color="navy", label="LSE")
        plt.xticks(np.arange(10), ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"])
        # plt.xlim(0,1)
        plt.title("Calibration")
        plt.xlabel("α")
        plt.ylabel("ECE")
        plt.grid(linewidth=0.3)
        plt.legend()
        plt.show()
