#! /usr/bin/python
# -*- coding: utf8 -*-
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto

from datetime import datetime
import scipy
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score

from deepsleepLite.data_loader import DataLoader
from deepsleepLite.model import SleepNetLite
from deepsleepLite.utils import *
from deepsleepLite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SAMPLING_RATE)

ALPHA = 0
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('alpha', 0.0,
                           """Directory where to save outputs.""")

tf.app.flags.DEFINE_string('data_dir', '/content/drive/MyDrive/Experiment _Paper/DOD-O_V2',
                           """Directory where to load testing data.""")
tf.app.flags.DEFINE_string('model_dir', '/content/output',
                           """Directory where to load trained models.""")
tf.app.flags.DEFINE_string('output_dir', f'/content/drive/MyDrive/Experiment _Paper/DSNL/DODO/predictionMC/LSE/1.0',
                           """Directory where to save outputs.""")

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
        fold_idx,
        MC_sampling

):
    start_time = time.time()
    y = []
    y_true = []
    y_selected = []
    y_true_selected = []

    y_var = []
    prob_pred = []
    prob_pred_selected = []

    query_instances = []
    correct_among_query = []
    all_prob = []

    total_loss, n_batches = 0.0, 0

    for sub_f_idx, each_data in enumerate(zip(inputs, targets, targets_smoothed)):

        each_x, each_y, each_y_cond = each_data

        each_y_true = []
        each_y_pred = []

        each_prob_pred = []
        each_y_var = []
        each_all_prob = []
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
            y_pred_tmp = np.empty((1, MC_sampling))
            prob_tmp = np.empty((MC_sampling, NUM_CLASSES))
            for n in range(0, MC_sampling):
                _, loss_value, y_pred, logits = sess.run(
                    [train_op, network.loss_op, network.pred_op, network.logits],
                    feed_dict=feed_dict)

                prob_tmp[n, :] = scipy.special.softmax(logits[0])
                y_pred_tmp[0, n] = y_pred
            each_all_prob.append(prob_tmp)
            mean_probs = np.mean(prob_tmp, axis=0)
            var_probs = np.var(prob_tmp, axis=0)

            y_pred = np.asarray([np.argmax(mean_probs)])

            each_y_true.extend(y_batch)
            each_y_pred.extend(y_pred)
            each_prob_pred.append(mean_probs)
            each_y_var.append(var_probs[y_pred[-1]])

            total_loss += loss_value
            n_batches += 1

            # Check the loss value
            assert not np.isnan(loss_value), \
                "Model diverged with loss = NaN"
        all_prob.append(each_all_prob)
        y.append(each_y_pred)
        y_true.append(each_y_true)
        prob_pred.append(each_prob_pred)
        y_var.append(each_y_var)

        n_examples = len(y_true[sub_f_idx])
        y_arr = np.asarray(y[sub_f_idx])
        y_true_arr = np.asarray(y_true[sub_f_idx])
        prob_pred_arr = np.asarray(prob_pred[sub_f_idx])
        y_var_arr = np.asarray(y_var[sub_f_idx])

        # Variance Rule selection - threshold 5% whole recording
        idx_threshold = int(0.95 * n_examples)
        # var_threshold = np.percentile(y_var_arr,90)
        var_threshold = np.sort(y_var_arr)[idx_threshold]
        removed_idx = np.where(y_var_arr >= var_threshold)[-1]
        selected_idx = np.where(y_var_arr < var_threshold)[-1]
        n_query = n_examples - len(selected_idx)

        query_instances.append(n_query)
        print('number of query {}'.format(n_query))
        correct_ = np.sum(y_true_arr[removed_idx] == y_arr[removed_idx])
        correct_among_query.append(correct_)
        print('number of which were correct {}'.format(correct_))

        y_selected.append(y_arr[selected_idx].tolist())
        y_true_selected.append(y_true_arr[selected_idx].tolist())
        prob_pred_selected.append(prob_pred_arr[selected_idx].tolist())

        # # Save prediction

    save_dict = {
        "y_true": y_true,
        "y_pred": y,
        "prob_pred": prob_pred,
        "y_var": y_var,
        "y_true_selected": y_true_selected,
        "y_pred_selected": y_selected,
        "prob_pred_selected": prob_pred_selected,
        "query_instances": query_instances,
        "correct_among_query": correct_among_query,
        "all_prob":all_prob
    }
    save_path = os.path.join(
        output_dir,
        "output_fold{}.npz".format(fold_idx)
    )

    np.savez(save_path, **save_dict)
    print("Saved outputs to {}".format(save_path))

    duration = time.time() - start_time
    total_loss /= n_batches
    total_y_pred = np.hstack(y)
    total_y_true = np.hstack(y_true)

    return total_y_true, total_y_pred, total_loss, duration


def predict_on_feature_net(
        data_dir,
        model_dir,
        output_dir,
        n_folds,
        smoothing,
        MC_dropout,
        MC_sampling
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
            use_MC_dropout=MC_dropout,
            freeze=False
        )

        # Initialize parameters
        test_net.init_ops()

        
        for fold_idx in range(n_folds):

            data_loader = DataLoader(
                data_dir=data_dir,
                n_folds=n_folds,
                fold_idx=fold_idx
            )

            checkpoint_path = f"{model_dir}/fold{fold_idx}/sleepnetlite/checkpoint/"
            test_path = f"{model_dir}/fold{fold_idx}/sleepnetlite/data_file{fold_idx}.npz"
            print({checkpoint_path})
            print({test_path})

            if not os.path.exists(checkpoint_path):
                Acc.append('NaN')
                continue

            # Restore the trained model
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
            print("Model restored from: {}\n".format(tf.train.latest_checkpoint(checkpoint_path)))

            # Load testing data -
            x, y, y_smoothed, test_files = data_loader.load_ISRC_testdata_cv(test_path)

            # Loop each epoch
            print("[{}] Predicting ...\n".format(datetime.now()))

            # # Evaluate the model on the subject data
            y_true_, y_pred_, loss, duration = \
                run_epoch(
                    sess=sess, network=test_net,
                    inputs=x, targets=y, targets_smoothed=y_smoothed,
                    seq_length=3,
                    smoothing=smoothing,
                    train_op=tf.no_op(),
                    output_dir=output_dir,
                    fold_idx=fold_idx,
                    MC_sampling=MC_sampling
                )

            n_examples = len(y_true)

            cm_ = confusion_matrix(y_true_, y_pred_)
            acc_ = np.mean(y_true_ == y_pred_)
            mf1_ = f1_score(y_true_, y_pred_, average="macro")
            k_ = cohen_kappa_score(y_true_, y_pred_)
            wf1_ = f1_score(y_true_, y_pred_, average="weighted")
            f1_ = f1_score(y_true_, y_pred_, average=None)


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
           

            y_true.extend(y_true_)
            y_pred.extend(y_pred_)

    # Overall performance

    print("[{}] Overall prediction performance\n".format(datetime.now()))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_examples = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    k = cohen_kappa_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    print((
        "n={}, acc={:.3f}, mf1={:.3f} wf1={:.3f} k={:.3f}".format(
            n_examples, acc, mf1, wf1, k
        )
    ))
    print((
        "Per-class-f1: w={:.3f}, n1={:.3f}, n2={:.3f}, n3={:.3f} , rem={:.3f}").format(
        per_class_f1[0], per_class_f1[1], per_class_f1[2], per_class_f1[3], per_class_f1[4]
    ))
    print(cm)
    save_dict = {
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "acc": acc,
        "mf1": mf1,
        "wf1": wf1,
        "k": k
    }
    print(f"alfa = {ALPHA}")
    np.savez(f"{output_dir}/performance_overall.npz", **save_dict)

    print(f" n={n_examples}, acc={round(np.mean(Acc)*100,1)} ± {round(np.std(Acc)*100,1)}, mf1={round(np.mean(MF1)*100,1)} ± {round(np.std(MF1)*100,1)}, wf1={round(np.mean(WF1)*100,1)} ± {round(np.std(WF1)*100,1)}, k={round(np.mean(K)*100,1)} ± {round(np.std(K)*100,1)}")
    print(f"Per-class-f1: w={round(np.mean(F1_w)*100,1)} ± {round(np.std(F1_w)*100,1)}, n1={round(np.mean(F1_n1)*100,1)} ± {round(np.std(F1_n1)*100,1)}, n2={round(np.mean(F1_n2)*100,1)} ± {round(np.std(F1_n2)*100,1)}, n3={round(np.mean(F1_n3)*100,1)} ± {round(np.std(F1_n3)*100,1)}, rem={round(np.mean(F1_r)*100,1)} ± {round(np.std(F1_r)*100,1)}")



def main(argv=None):
    # Output dir
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    n_folds = 10

    predict_on_feature_net(
        data_dir=FLAGS.data_dir,
        model_dir=FLAGS.model_dir,
        output_dir=FLAGS.output_dir,
        n_folds=n_folds,
        smoothing=FLAGS.alpha,
        MC_dropout=True,
        MC_sampling=50
    )


if __name__ == "__main__":
    tf.compat.v1.app.run()
