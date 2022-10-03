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
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from deepsleepLite.data_loader import DataLoader
from deepsleepLite.model import SleepNetLite
from deepsleepLite.utils import *
from deepsleepLite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SAMPLING_RATE)
from ece_acs import*
from tabulate import tabulate


# Ignore os, tf depecration errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

dataset = sys.argv[1]
model = sys.argv[2]
base_path = "/content/drive/MyDrive/Experiments/DSNL"
model_dir = f"{base_path}/{dataset}/{model}"
n_folds = 1
alpha = 0

if dataset == "DODO":
  data_dir = "/content/drive/MyDrive/Experiments/data/DODO"
  fold_idx = 1
elif dataset == "DODH":
  data_dir = "/content/drive/MyDrive/Experiments/data/DODH"
  fold_idx = 24
elif dataset == "ISRC":
  data_dir = "/content/drive/MyDrive/Experiments/data/ISRC"
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
        fold_idx

):
    start_time = time.time()
    y = []
    y_true = []
    prob_pred = []
    ece = []
    acs = []
    acc = []
    mf1 = []
    wf1 = []
    k = []
    f1_w = []
    f1_n1 = []
    f1_n2 = []
    f1_n3 = []
    f1_r = []
    all_y_true = []
    all_y_pred = []
    all_hypno_true = []
    all_hypno_pred = []
    

    total_loss, n_batches = 0.0, 0

    for sub_f_idx, each_data in enumerate(zip(inputs, targets, targets_smoothed)):
        print(".", end="",flush=True)
        each_x, each_y, each_y_cond = each_data

        each_y_true = []
        each_y_pred = []
        each_hypno_true = []
        each_hypno_pred = []
        each_conf_pred = []


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
            each_hypno_true.append(y_batch_cond[0])
            each_hypno_pred.append(prob_tmp)

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"

        y.append(each_y_pred)
        y_true.append(each_y_true)
        prob_pred.append(each_hypno_pred)

        # Compute ECE
        each_y_true = [int(i) for i in each_y_true]
        ece.append(compute_calibration(np.array(each_y_true), np.array(each_y_pred), np.array(each_conf_pred), num_bins=20))

        # Compute ACS
        acs.append(compute_acs(each_hypno_true, each_hypno_pred))

        # Compute Performance
        acc.append(np.mean(accuracy_score(each_y_true, each_y_pred)))
        mf1.append(f1_score(each_y_true, each_y_pred, average="macro"))
        k.append(cohen_kappa_score(each_y_true, each_y_pred))
        wf1.append(f1_score(each_y_true, each_y_pred, average="weighted"))
        f1_ = f1_score(each_y_true, each_y_pred, average=None)
        cls = np.setdiff1d(np.array([0,1,2,3,4]),np.unique(each_y_pred))
        # Check if there is at least one example for each class
        d = {}
        for n,i in enumerate(np.unique(each_y_pred)):
          d[i] = f1_[n]
        if cls.size > 0:
          for i in cls:
            d[i] = np.nan
        f1_w.append(d[0])
        f1_n1.append(d[1])
        f1_n2.append(d[2])
        f1_n3.append(d[3])
        f1_r.append(d[4])

        # Storing all date to save
        all_y_true.append(each_y_true)
        all_y_pred.append(each_y_pred)
        all_hypno_true.append(each_hypno_true)
        all_hypno_pred.append(each_hypno_pred)



    duration = np.round(time.time() - start_time,1)
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return duration, acc, mf1, k, wf1, f1_w, f1_n1, f1_n2, f1_n3, f1_r, ece, acs, all_y_true, all_y_pred, all_hypno_true, all_hypno_pred


def predict_on_feature_net(
        data_dir,
        model_dir,
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

        print(f"Checkpoint_path: {checkpoint_path} \n")  
                  
        print(f"Test_path: {test_path} \n")

        if not os.path.exists(checkpoint_path):
            Acc.append('NaN')
            
        # Restore the trained model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        print(f"Model restored from: {tf.train.latest_checkpoint(checkpoint_path)}\n")

        # Load testing data -
        x, y, y_smoothed, test_files = data_loader.load_testdata_cv(test_path)

        print(f"Patients predicted: {test_files} \n")

        # Loop each epoch
        print(f"[{datetime.now()}] Predicting ",end="",flush=True)

        # Evaluate the model on the subject data
        duration, acc, mf1, k, wf1, f1_w, f1_n1, f1_n2, f1_n3, f1_r, ece, acs, all_y_true, all_y_pred, all_hypno_true, all_hypno_pred = \
            run_epoch(
                sess=sess, network=test_net,
                inputs=x, targets=y, targets_smoothed=y_smoothed,
                seq_length=3,
                smoothing=smoothing,
                train_op=tf.no_op(),
                fold_idx=fold_idx
            )
        print(f"Done! [Time elapsed: {duration} s]")


    # Overall performance
    Acc = np.round(np.mean(acc)*100,1)
    MF1 = np.round(np.mean(mf1)*100,1)
    K = np.round(np.mean(k)*100,1)
    WF1 = np.round(np.mean(wf1)*100,1)
    F1_w = np.round(np.nanmean(f1_w)*100,1)
    F1_n1 = np.round(np.nanmean(f1_n1)*100,1)
    F1_n2 = np.round(np.nanmean(f1_n2)*100,1)
    F1_n3 = np.round(np.nanmean(f1_n3)*100,1)
    F1_r = np.round(np.nanmean(f1_r)*100,1)
    
    acc_ = np.round(np.mean(acc),3)
    conf = []
    for k in ece:
      conf.append(k['avg_confidence'])
    conf = np.round(np.mean(conf),3)
    ece_ = round(abs(acc_ - conf),3)
    acs = f"{np.round(np.mean(acs),3)} ± {np.round(np.std(acs),3)}"

    print("\nOverall Performance Tables: \n")
    print(tabulate([[dataset, f"DSNL {model}", Acc, MF1, WF1, K, F1_w, F1_n1, F1_n2, F1_n3, F1_r]], headers=['Dataset','Model','Accuracy %', 'MF1 %', 'WF1 %','Cohen-k %', 'W %', 'N1 %', 'N2 %','N3 %','REM %'], tablefmt="pretty"))
    print(tabulate([[dataset, f"DSNL {model}", ece_, acc_, conf, acs]], headers=['Dataset','Model','ECE', 'Accuracy', 'Confidence','ACS'],tablefmt="pretty"))

  
    # Saving Prediction
    save_dict = {
      "y_true" : all_y_true,
      "y_pred":all_y_pred,
      "hypno_true":all_hypno_true,
      "hypno_pred":all_hypno_pred
    }

    np.savez(f"/content/drive/MyDrive/Experiments/plot_data/DSNL/output_fold{fold_idx}_{dataset}_{model}.npz", **save_dict)
    print(f"Prediction saved to path /content/drive/MyDrive/Experiments/plot_data/DSNL/output_fold{fold_idx}_{dataset}_{model}.npz")

def main(argv=None):

    predict_on_feature_net(
        data_dir=data_dir,
        model_dir=model_dir,
        n_folds=n_folds,
        smoothing=alpha
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
