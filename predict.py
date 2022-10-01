#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import time
import sys
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto

from datetime import datetime
import scipy
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, precision_score, recall_score

from deepsleepLite.data_loader import DataLoader
from deepsleepLite.model import SleepNetLite
from deepsleepLite.utils import *
from deepsleepLite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SAMPLING_RATE)
from Calibration_ACS import*
from tabulate import tabulate


# Ignore os, tf depecration errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dataset = sys.argv[1]
model = sys.argv[2]
base_path = "/content/drive/MyDrive/Experiments/DSNL"
model_dir = f"{base_path}/{dataset}/{model}"
output_dir = "/content/pred"
n_folds = 1
alpha = 0

if dataset == "DODO":
  data_dir = "/content/drive/MyDrive/Experiment _Paper/DOD-O_V2"
  fold_idx = 1
elif dataset == "DODH":
  data_dir = "/content/drive/MyDrive/Experiment _Paper/DOD-H_V2"
  fold_idx = 24
elif dataset == "ISRC":
  data_dir = "/content/drive/MyDrive/Experiment _Paper/IS-RC_filtered_one+smooth_correct"
  fold_idx = 4

coding2stages = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R"
}


def print_performance(sess, network_name, n_examples, duration, loss, cm, acc, f1):
    # Get regularization loss
    reg_loss = tf.add_n(tf.get_collection("losses", scope=network_name + "\/"))
    reg_loss_value = sess.run(reg_loss)

    # Print performance
    print((
        "duration={:.3f} sec, n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
        "f1={:.3f}".format(
            duration, n_examples, loss, reg_loss_value, acc, f1
        )
    ))
    print(cm)
    print(" ")


def run_epoch(
        sess,
        network,
        inputs,
        targets,
        targets_smoothed,
        seq_length,
        smoothing,
        train_op,
        output_dir,
        fold_idx

):
    start_time = time.time()
    y = []
    y_true = []
    prob_pred = []
    ece = []
    acs = []
    

    total_loss, n_batches = 0.0, 0

    for sub_f_idx, each_data in enumerate(zip(inputs, targets, targets_smoothed)):

        each_x, each_y, each_y_cond = each_data

        each_y_true = []
        each_y_pred = []
        each_hypno_sc = []
        each_conf_pred = []
        
       

        each_prob_pred = []
        # y_batch_seq, batch_len, epochs_shifts
        for x_batch, y_batch, y_batch_cond in iterate_minibatches_prediction(
                inputs=each_x,
                targets=each_y,
                targets_cond=each_y_cond,
                batch_size=network.batch_size,
                seq_length=seq_length):

            feed_dict = {
                network.input_var: x_batch,
                network.target_var: y_batch,
                network.target_var_conditioned: y_batch_cond,
                network.alfa: smoothing,
            }

            _, loss_value, y_pred, logits = sess.run(
                [train_op, network.loss_op, network.pred_op, network.logits],
                feed_dict=feed_dict)

            prob_tmp = scipy.special.softmax(logits[0])
            y_pred = np.asarray([np.argmax(prob_tmp)])

            each_y_true.extend(y_batch)
            each_y_pred.extend(y_pred)
            each_conf_pred.extend([np.max(prob_tmp)])
            each_hypno_sc.append(y_batch_cond[0])
            each_prob_pred.append(prob_tmp)

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        y.append(each_y_pred)
        y_true.append(each_y_true)
        prob_pred.append(each_prob_pred)

        # Compute ECE
        each_y_true = [int(i) for i in each_y_true]
        ece.append(compute_calibration(np.array(each_y_true), np.array(each_y_pred), np.array(each_conf_pred), num_bins=20))

        # Compute ACS
        acs.append(compute_similarity(each_prob_pred, each_hypno_sc))


    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration, ece, acs


def predict_on_feature_net(
        data_dir,
        model_dir,
        output_dir,
        n_folds,
        smoothing
):
    # Ground truth and predictions
    y_true = []
    y_pred = []

    Acc = []
    WF1 = []
    K = []
    MF1 = []
    F1_w = []
    F1_n1 = []
    F1_n2 = []
    F1_n3 = []
    F1_r = []
    Pre = []
    Rec = []



    # The model will be built into the default Graph
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        test_net = SleepNetLite(
            batch_size=1,
            input_dims=EPOCH_SEC_LEN * SAMPLING_RATE,
            seq_length=3,
            n_classes=NUM_CLASSES,
            is_train=False,
            reuse_params=False,
            use_MC_dropout=False,
            freeze=False
        )

        # Initialize parameters
        test_net.init_ops()

    
        data_loader = DataLoader(
            data_dir=data_dir,
            n_folds=n_folds,
            fold_idx=fold_idx
        )

        checkpoint_path = f"{model_dir}/fold{fold_idx}/sleepnetlite/checkpoint/"
        test_path = f"{model_dir}/fold{fold_idx}/sleepnetlite/data_file{fold_idx}.npz"

        print(f"checkpoint_path: {checkpoint_path} \n")  
                  
        print(f"test_path: {test_path} \n")

        if not os.path.exists(checkpoint_path):
            Acc.append('NaN')
            
        # Restore the trained model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

        # Load testing data -
        x, y, y_smoothed, test_files = data_loader.load_ISRC_testdata_cv(test_path)

        print(f"Patient predicted: {test_files} \n")

        # Loop each epoch
        print("[{}] Predicting ...\n".format(datetime.now()))

        # # Evaluate the model on the subject data
        y_true_, y_pred_, loss, duration, ece, acs = \
            run_epoch(
                sess=sess, network=test_net,
                inputs=x, targets=y, targets_smoothed=y_smoothed,
                seq_length=3,
                smoothing=smoothing,
                train_op=tf.no_op(),
                output_dir=output_dir,
                fold_idx=fold_idx
            )

        n_examples = len(y_true)

        cm_ = confusion_matrix(y_true_, y_pred_)
        acc_ = np.mean(y_true_ == y_pred_)
        mf1_ = f1_score(y_true_, y_pred_, average="macro")
        k_ = cohen_kappa_score(y_true_, y_pred_)
        wf1_ = f1_score(y_true_, y_pred_, average="weighted")
        f1_ = f1_score(y_true_, y_pred_, average=None)
        # pre_ = precision_score(y_true_, y_pred_)
        # rec_ = recall_score(y_true_, y_pred_)

        save_dict = {
            "test_files": test_files,
            "cm": cm_,
            "acc": acc_,
            "mf1": mf1_
        }
        np.savez(f"{output_dir}/performance_fold{fold_idx}.npz", **save_dict)

        # Report performance
        print_performance(
            sess, test_net.name,
            n_examples, duration, loss,
            cm_, acc_, mf1_
        )

        # All folds
        Acc.append(acc_)
        MF1.append(mf1_)
        WF1.append(wf1_)
        K.append(k_)
        F1_w.append(f1_[0])
        F1_n1.append(f1_[1])
        F1_n2.append(f1_[2])
        F1_n3.append(f1_[3])
        F1_r.append(f1_[4])
        # Pre.append(pre_)
        # Rec.append(rec_)

        y_true.extend(y_true_)
        y_pred.extend(y_pred_)

    # Overall performance
    print("[{}] Overall prediction performance\n".format(datetime.now()))


    Acc = np.round(np.mean(Acc)*100,1)
    MF1 = np.round(np.mean(MF1)*100,1)
    WF1 = np.round(np.mean(WF1)*100,1)
    K = np.round(np.mean(K)*100,1)
    F1_w = np.round(np.mean(F1_w)*100,1)
    F1_n1 = np.round(np.mean(F1_n1)*100,1)
    F1_n2 = np.round(np.mean(F1_n2)*100,1)
    F1_n3 = np.round(np.mean(F1_n3)*100,1)
    F1_r = np.round(np.mean(F1_r)*100,1)

    ece__ = []
    acc__ = []
    conf__ = []
    for k in ece:
      ece__.append(k['expected_calibration_error'])
      acc__.append(k['avg_accuracy'])
      conf__.append(k['avg_confidence'])
    ece__ = np.round(np.mean(ece__),3)
    acc__ = np.round(np.mean(acc__),4)
    conf__ = np.round(np.mean(conf__),4)
    acs = f"{np.round(np.mean(acs),3)} ± {np.round(np.std(acs),3)}"


    print(tabulate([[dataset, f"DSNL {model}", Acc, MF1, WF1, K, F1_w, F1_n1, F1_n2, F1_n3, F1_r]], headers=['Dataset','Model','Accuracy %', 'MF1 %', 'WF1 %','Cohen-k %', 'W %', 'N1 %', 'N2 %','N3 %','REM %'], tablefmt="pretty"))
    print(tabulate([[dataset, f"DSNL {model}", ece__, acc__, conf__, acs]], headers=['Dataset','Model','ECE', 'Accuracy', 'Confidence','ACS'],tablefmt="pretty"))


def main(argv=None):
    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    

    predict_on_feature_net(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        n_folds=n_folds,
        smoothing=alpha
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
