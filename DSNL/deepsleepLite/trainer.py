import os
import time

import matplotlib

matplotlib.use("Agg")

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto

from sklearn.metrics import confusion_matrix, f1_score
from datetime import datetime

from deepsleepLite.data_loader import DataLoader
from deepsleepLite.model import SleepNetLite
from deepsleepLite.optimize import adam
from deepsleepLite.utils import *


class Trainer(object):

    def __init__(self, interval_print_cm=5):
        self.interval_print_cm = interval_print_cm

    def print_performance(self, sess, output_dir, network_name,
                          n_train_examples, n_valid_examples,
                          train_cm, valid_cm, epoch, n_epochs,
                          train_duration, train_loss, train_acc, train_f1,
                          valid_duration, valid_loss, valid_acc, valid_f1):
        # Get regularization loss
        train_reg_loss = tf.add_n(tf.compat.v1.get_collection("losses", scope=network_name + "\/"))
        train_reg_loss_value = sess.run(train_reg_loss)
        valid_reg_loss_value = train_reg_loss_value

        # Print performance
        if ((epoch + 1) % self.interval_print_cm == 0) or ((epoch + 1) == n_epochs):
            print(" ")
            print("[{}] epoch {}:".format(
                datetime.now(), epoch + 1
            ))
            print((
                "train ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}".format(
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1
                )
            ))
            print(train_cm)
            print((
                "valid ({:.3f} sec): n={}, loss={:.3f} ({:.3f}), acc={:.3f}, "
                "f1={:.3f}".format(
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1
                )
            ))
            print(valid_cm)
            print(" ")
        else:
            print((
                "epoch {}: "
                "train ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f} | "
                "valid ({:.2f} sec): n={}, loss={:.3f} ({:.3f}), "
                "acc={:.3f}, f1={:.3f}".format(
                    epoch + 1,
                    train_duration, n_train_examples,
                    train_loss, train_reg_loss_value,
                    train_acc, train_f1,
                    valid_duration, n_valid_examples,
                    valid_loss, valid_reg_loss_value,
                    valid_acc, valid_f1
                )
            ))

    def print_network(self, network):
        print("inputs ({}): {}".format(
            network.inputs.name, network.inputs.get_shape()
        ))
        print("targets ({}): {}".format(
            network.targets.name, network.targets.get_shape()
        ))
        for name, act in network.activations:
            print("{} ({}): {}".format(name, act.name, act.get_shape()))
        print(" ")


class SleepNetLiteTrainer(Trainer):

    def __init__(
            self,
            data_dir,
            output_dir,
            n_folds,
            fold_idx,
            batch_size,
            input_dims,
            seq_length,
            n_classes,
            interval_print_cm=5
    ):
        super(self.__class__, self).__init__(
            interval_print_cm=interval_print_cm
        )

        self.data_dir = data_dir
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.seq_length = seq_length
        self.n_classes = n_classes


    def _run_epoch_tinyDB(self, sess, network, data, targets, targets_smoothed, train_op, is_train, writer, freeze, balance_classes,
                          smoothing):

        start_time = time.time()
        y = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        is_shuffle = True if is_train else False

        if is_train:

            # Balance sleep stages classes
            if balance_classes == 'oversampling':
                data, targets, targets_smoothed = get_balance_class_sequences_oversample(x=data, y=targets, y_smoothed=targets_smoothed, seq_length=self.seq_length,
                                                                       flipping=True, cond_prob=True) 

            for x_batch, y_batch, y_batch_seq, y_batch_conditioned in iterate_minibatches_train( 
                    np.reshape(data, (-1, self.seq_length * int(self.input_dims), 1, 1)), 
                    targets, 
                    targets_smoothed, 
                    self.batch_size,
                    self.seq_length,
                    shuffle=is_shuffle):

                feed_dict = {
                    network.input_var: x_batch,
                    network.target_var: y_batch,
                    network.target_var_conditioned: y_batch_conditioned,
                    network.alfa: smoothing
                }

                # Run network
                _, loss_value, y_pred = sess.run(
                    [train_op, network.loss_op, network.pred_op],
                    feed_dict=feed_dict)

                total_loss += loss_value
                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)

                # Check the loss value
                assert not np.isnan(loss_value), \
                    "Model diverged with loss = NaN"

        else:

            for x_batch, y_batch, y_batch_seq, y_batch_conditioned in iterate_minibatches_valid(
                    np.reshape(data, (-1, self.seq_length * int(self.input_dims), 1, 1)),
                    targets,
                    targets_smoothed,  
                    self.batch_size,
                    self.seq_length,
                    shuffle=is_shuffle):


                feed_dict = {
                    network.input_var: x_batch,
                    network.target_var: y_batch,
                    network.target_var_conditioned: y_batch_conditioned,
                    network.alfa: smoothing
                }

                _, loss_value, y_pred = sess.run(
                    [train_op, network.loss_op, network.pred_op],
                    feed_dict=feed_dict)

                total_loss += loss_value
                n_batches += 1
                y.append(y_pred)
                y_true.append(y_batch)

                # Check the loss value
                assert not np.isnan(loss_value), \
                    "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss, duration

    def _run_epoch(self, sess, network, inputs, data_loader, train_op, is_train, writer, freeze, balance_classes,
                   smoothing):

        start_time = time.time()
        y = []
        y_true = []
        total_loss, n_batches = 0.0, 0
        is_shuffle = True if is_train else False
        batch_files_size = 50 if is_train else 1

        for i in range(0, len(inputs) - batch_files_size + 1, batch_files_size):

            # Extract batch files from all the files in input
            if i + (2 * batch_files_size) > len(inputs):
                batch_files_inputs = inputs[i:]
            else:
                batch_files_inputs = inputs[i:i + batch_files_size]

            if is_train:

                # Load data input sequences from files
                data, targets, sampling_rate = data_loader.load_data_sequences(input_files=batch_files_inputs,
                                                                                     seq_length=self.seq_length, train=True)

                # Balance sleep stages classes
                if balance_classes == 'oversampling':
                    data, targets, indices_list = get_balance_class_sequences_oversample(x=data, y=targets,
                                                                           seq_length=self.seq_length, flipping=True)

                for x_batch, y_batch, y_batch_seq in iterate_minibatches_train(
                        np.reshape(data, (-1, self.seq_length * int(self.input_dims), 1, 1)),  
                        targets,
                        self.batch_size,
                        self.seq_length,
                        shuffle=is_shuffle):

                    feed_dict = {
                        network.input_var: x_batch,
                        network.target_var: y_batch,
                        network.alfa: smoothing
                    }

                    _, loss_value, y_pred = sess.run(
                        [train_op, network.loss_op, network.pred_op],
                        feed_dict=feed_dict)

                    total_loss += loss_value
                    n_batches += 1
                    y.append(y_pred)
                    y_true.append(y_batch)

                    # Check the loss value
                    assert not np.isnan(loss_value), \
                        "Model diverged with loss = NaN"

            else:

                # Load data input sequences from files
                data, targets, sampling_rate = data_loader.load_data_sequences(input_files=batch_files_inputs,
                                                                                   seq_length=self.seq_length, train=False)

                for x_batch, y_batch, y_batch_seq in iterate_minibatches_valid(
                        np.reshape(data, (-1, self.seq_length * int(self.input_dims), 1, 1)),
                        targets,
                        self.batch_size,
                        self.seq_length,
                        shuffle=is_shuffle):

                    feed_dict = {
                        network.input_var: x_batch,
                        network.target_var: y_batch,
                        network.alfa: smoothing
                    }

                    _, loss_value, y_pred = sess.run(
                        [train_op, network.loss_op, network.pred_op],
                        feed_dict=feed_dict)

                    total_loss += loss_value
                    n_batches += 1
                    y.append(y_pred)
                    y_true.append(y_batch)

                    # Check the loss value
                    assert not np.isnan(loss_value), \
                        "Model diverged with loss = NaN"

        duration = time.time() - start_time
        total_loss /= n_batches
        total_y_pred = np.hstack(y)
        total_y_true = np.hstack(y_true)

        return total_y_true, total_y_pred, total_loss, duration

    def train(self, n_epochs, resume, freeze, alpha):

        # Limit GPU memory Usage
        config = ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Graph().as_default(), tf.compat.v1.Session(config=config) as sess:
            # Build training and validation networks
            train_net = SleepNetLite(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                seq_length=self.seq_length,
                n_classes=self.n_classes,
                is_train=True,
                reuse_params=False,
                use_MC_dropout=False,
                freeze=freeze
            )
            valid_net = SleepNetLite(
                batch_size=self.batch_size,
                input_dims=self.input_dims,
                seq_length=self.seq_length,
                n_classes=self.n_classes,
                is_train=False,
                reuse_params=True,
                use_MC_dropout=False,
                freeze=freeze
            )

            # Initialize parameters
            train_net.init_ops()
            valid_net.init_ops()

            # Total number of trainable parameters
            trainable_parameters = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
            print("Total number of trainable parameters : {}".format(int(trainable_parameters)))

            print("Network (layers={})".format(len(train_net.activations)))
            print("inputs ({}): {}".format(
                train_net.input_var.name, train_net.input_var.get_shape()
            ))
            print("targets ({}): {}".format(
                train_net.target_var.name, train_net.target_var.get_shape()
            ))
            for name, act in train_net.activations:
                print("{} ({}): {}".format(name, act.name, act.get_shape()))
            print(" ")

            # Define optimization operations
            train_op, grads_and_vars_op = adam(
                loss=train_net.loss_op,
                lr=1e-4,
                train_vars=tf.compat.v1.trainable_variables()
            )

            # Make subdirectory for pretraining
            output_dir = os.path.join(self.output_dir, "fold{}".format(self.fold_idx), train_net.name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Define checkpoint directory
            ckpt_dir = os.path.join(output_dir, "checkpoint")

            # Global step for resume training
            with tf.compat.v1.variable_scope(train_net.name) as scope:
                global_step = tf.Variable(0, name="global_step", trainable=False)

            # Create a saver
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=0)

            # Initialize variables in the graph
            sess.run(tf.compat.v1.global_variables_initializer())

            # Add the graph structure into the Tensorboard writer
            train_summary_wrt = tf.compat.v1.summary.FileWriter(
                os.path.join(output_dir, "train_summary"))
            train_summary_wrt.add_graph(sess.graph)

            # Resume the training if applicable
            if resume:

                if os.path.exists(output_dir):
                    if os.path.isdir(os.path.join(output_dir, "checkpoint")):
                        # Restore the last checkpoint
                        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
                        # saver.restore(sess, tf.train.latest_checkpoint(output_dir))
                        print("Model restored")
                        print("[{}] Resume pre-training ...\n".format(datetime.now()))
                    else:
                        print("[{}] Start pre-training ...\n".format(datetime.now()))
            else:

                print("[{}] Start pre-training ...\n".format(datetime.now()))

            check_point_global_step = sess.run(global_step)

            # Load data
            if sess.run(global_step) < n_epochs:

                # Data Loader in many_to_one test case
                data_loader = DataLoader(
                    data_dir=self.data_dir,
                    n_folds=self.n_folds,
                    fold_idx=self.fold_idx
                )

                if resume:

                    with np.load(os.path.join(output_dir, "data_file{}.npz".format(self.fold_idx))) as f:
                        train_files = f["train_files"]
                        valid_files = f["valid_files"]
                        test_files = f["test_files"]

                else:

                    # Load files:
                    train_files, valid_files, test_files = data_loader.load_files_cv(dodo=True)

                    # Load data input sequences from files - tinyDB
                    data_train, targets_train, sampling_rate, targets_prob_train = data_loader.load_data_sequences(
                        input_files=train_files, seq_length=self.seq_length, train=True, cond_prob=True)
                    data_valid, targets_valid, sampling_rate, targets_prob_valid = data_loader.load_data_sequences(
                        input_files=valid_files, seq_length=self.seq_length, train=False, cond_prob=True)

                # Performance history
                all_train_loss = np.zeros(n_epochs)
                all_train_acc = np.zeros(n_epochs)
                all_train_f1 = np.zeros(n_epochs)
                all_train_per_class_precision = np.zeros((n_epochs, self.n_classes))
                all_train_per_class_recall = np.zeros((n_epochs, self.n_classes))
                all_train_per_class_f1 = np.zeros((n_epochs, self.n_classes))
                all_train_mf1 = np.zeros(n_epochs)
                all_valid_loss = np.zeros(n_epochs)
                all_valid_acc = np.zeros(n_epochs)
                all_valid_f1 = np.zeros(n_epochs)
                all_valid_per_class_precision = np.zeros((n_epochs, self.n_classes))
                all_valid_per_class_recall = np.zeros((n_epochs, self.n_classes))
                all_valid_per_class_f1 = np.zeros((n_epochs, self.n_classes))
                all_valid_mf1 = np.zeros(n_epochs)

                if resume:

                    with np.load(os.path.join(output_dir, "perf_fold{}.npz".format(self.fold_idx))) as f:

                        train_loss = f["train_loss"]
                        train_acc = f["train_acc"]
                        train_f1 = f["train_f1"]
                        train_per_class_precision = f["train_per_class_precision"]
                        train_per_class_recall = f["train_per_class_recall"]
                        train_per_class_f1 = f["train_per_class_f1"]
                        train_mf1 = f["train_mf1"]
                        valid_loss = f["valid_loss"]
                        valid_acc = f["valid_acc"]
                        valid_f1 = f["valid_f1"]
                        valid_per_class_precision = f["valid_per_class_precision"]
                        valid_per_class_recall = f["valid_per_class_recall"]
                        valid_per_class_f1 = f["valid_per_class_f1"]
                        valid_mf1 = f["valid_mf1"]

                        all_train_loss[:check_point_global_step] = train_loss[:check_point_global_step]
                        all_train_acc[:check_point_global_step] = train_acc[:check_point_global_step]
                        all_train_f1[:check_point_global_step] = train_f1[:check_point_global_step]
                        all_train_per_class_precision[:check_point_global_step, :] = train_per_class_precision[
                                                                                     :check_point_global_step, :]
                        all_train_per_class_recall[:check_point_global_step, :] = train_per_class_recall[
                                                                                  :check_point_global_step, :]
                        all_train_per_class_f1[:check_point_global_step, :] = train_per_class_f1[
                                                                              :check_point_global_step, :]
                        all_train_mf1[:check_point_global_step] = train_mf1[:check_point_global_step]
                        all_valid_loss[:check_point_global_step] = valid_loss[:check_point_global_step]
                        all_valid_acc[:check_point_global_step] = valid_acc[:check_point_global_step]
                        all_valid_f1[:check_point_global_step] = valid_f1[:check_point_global_step]
                        all_valid_per_class_precision[:check_point_global_step, :] = valid_per_class_precision[
                                                                                     :check_point_global_step, :]
                        all_valid_per_class_recall[:check_point_global_step, :] = valid_per_class_recall[
                                                                                  :check_point_global_step, :]
                        all_valid_per_class_f1[:check_point_global_step, :] = valid_per_class_f1[
                                                                              :check_point_global_step, :]
                        all_valid_mf1[:check_point_global_step] = valid_mf1[:check_point_global_step]
                        best_valid_acc = valid_acc[check_point_global_step - 1]

                else:

                    best_valid_acc = 0

                patience = 100
                patience_cnt = 0

            start = time.time()

            # Loop each epoch

            for epoch in range(sess.run(global_step), n_epochs):

                # random.shuffle(train_files)

                # Update parameters and compute loss of training set

                # tiny DB
                y_true_train, y_pred_train, train_loss, train_duration = \
                    self._run_epoch_tinyDB(
                        sess=sess, network=train_net,
                        data=data_train,
                        targets=targets_train,
                        targets_smoothed=targets_prob_train,
                        train_op=train_op,
                        is_train=True,
                        writer=train_summary_wrt,
                        freeze=freeze,
                        balance_classes='oversampling',
                        smoothing=alpha
                    )

                n_train_examples = len(y_true_train)
                train_cm = confusion_matrix(y_true_train, y_pred_train, labels=[0, 1, 2, 3, 4])
                train_acc = np.mean(y_true_train == y_pred_train)
                train_f1 = f1_score(y_true_train, y_pred_train, average="weighted")

                tp = np.diagonal(train_cm).astype(np.float)
                tpfp = np.sum(train_cm, axis=0).astype(np.float)  # sum of each col
                tpfn = np.sum(train_cm, axis=1).astype(np.float)  # sum of each row
                train_per_class_precision = tp / tpfp
                train_per_class_recall = tp / tpfn
                train_per_class_f1 = (2 * train_per_class_precision * train_per_class_recall) / (
                            train_per_class_precision + train_per_class_recall)
                train_mf1 = np.mean(train_per_class_f1)

                # tiny DB
                y_true_valid, y_pred_valid, valid_loss, valid_duration = \
                    self._run_epoch_tinyDB(
                        sess=sess, network=valid_net,
                        data=data_valid,
                        targets=targets_valid,
                        targets_smoothed=targets_prob_valid,
                        train_op=tf.no_op(),
                        is_train=False,
                        writer=train_summary_wrt,
                        freeze=freeze,
                        balance_classes='oversampling',
                        smoothing=alpha
                    )


                n_valid_examples = len(y_true_valid)
                valid_cm = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1, 2, 3, 4])
                valid_acc = np.mean(y_true_valid == y_pred_valid)
                valid_f1 = f1_score(y_true_valid, y_pred_valid, average="weighted")

                tp = np.diagonal(valid_cm).astype(np.float)
                tpfp = np.sum(valid_cm, axis=0).astype(np.float)  # sum of each col
                tpfn = np.sum(valid_cm, axis=1).astype(np.float)  # sum of each row
                valid_per_class_precision = tp / tpfp
                valid_per_class_recall = tp / tpfn
                valid_per_class_f1 = (2 * valid_per_class_precision * valid_per_class_recall) / (
                            valid_per_class_precision + valid_per_class_recall)
                valid_mf1 = np.mean(valid_per_class_f1)

                all_train_loss[epoch] = train_loss
                all_train_acc[epoch] = train_acc
                all_train_f1[epoch] = train_f1
                all_train_per_class_precision[epoch, :] = train_per_class_precision
                all_train_per_class_recall[epoch, :] = train_per_class_recall
                all_train_per_class_f1[epoch, :] = train_per_class_f1
                all_train_mf1[epoch] = train_mf1
                all_valid_loss[epoch] = valid_loss
                all_valid_acc[epoch] = valid_acc
                all_valid_f1[epoch] = valid_f1
                all_valid_per_class_precision[epoch, :] = valid_per_class_precision
                all_valid_per_class_recall[epoch, :] = valid_per_class_recall
                all_valid_per_class_f1[epoch, :] = valid_per_class_f1
                all_valid_mf1[epoch] = valid_mf1

                # Report performance
                self.print_performance(
                    sess, output_dir, train_net.name,
                    n_train_examples, n_valid_examples,
                    train_cm, valid_cm, epoch, n_epochs,
                    train_duration, train_loss, train_acc, train_f1,
                    valid_duration, valid_loss, valid_acc, valid_f1
                )

                # Save performance history
                np.savez(
                    os.path.join(output_dir, "perf_fold{}.npz".format(self.fold_idx)),
                    train_loss=all_train_loss, valid_loss=all_valid_loss,
                    train_acc=all_train_acc, valid_acc=all_valid_acc,
                    train_f1=all_train_f1, valid_f1=all_valid_f1,
                    train_per_class_precision=all_train_per_class_precision,
                    valid_per_class_precision=all_valid_per_class_precision,
                    train_per_class_recall=all_train_per_class_recall,
                    valid_per_class_recall=all_valid_per_class_recall,
                    train_per_class_f1=all_train_per_class_f1, valid_per_class_f1=all_valid_per_class_f1,
                    train_mf1=all_train_mf1, valid_mf1=all_valid_mf1,
                    y_true_valid=np.asarray(y_true_valid),
                    y_pred_valid=np.asarray(y_pred_valid)
                )

                sess.run(tf.compat.v1.assign(global_step, epoch + 1))

                diff_valid_acc = valid_acc - best_valid_acc
                if diff_valid_acc < 0:
                    patience_cnt += 1
                    if patience_cnt > patience:

                        print("....................early stopping....................")
                        break
                    else:
                        pass
                else:
                    best_valid_acc = valid_acc
                    start_time = time.time()
                    if os.path.exists(ckpt_dir):
                        tf.gfile.DeleteRecursively(ckpt_dir)
                    os.makedirs(ckpt_dir)
                    save_path = os.path.join(
                        ckpt_dir, "model_fold{}.ckpt".format(self.fold_idx)
                    )
                    saver.save(sess, save_path, global_step=global_step)
                    duration = time.time() - start_time
                    print("Saved model checkpoint ({:.3f} sec)".format(duration))
                    save_dict = {}
                    for v in tf.compat.v1.global_variables():
                        save_dict[v.name] = sess.run(v)
                    np.savez(
                        os.path.join(
                            output_dir,
                            "params_fold{}.npz".format(self.fold_idx)),
                        **save_dict
                    )
                    duration = time.time() - start_time
                    print("Saved trained parameters ({:.3f} sec)".format(duration))

                    patience_cnt = 0
                    pass

            if not resume:

                # Save training set and validation set files
                data_file_path = os.path.join(output_dir, "data_file{}.npz".format(self.fold_idx))

                if not os.path.exists(data_file_path):
                    save_dict = {
                        "train_files": train_files, "valid_files": valid_files, "test_files": test_files
                    }
                    np.savez(data_file_path, **save_dict)

        print("Finish pre-training")
        np.savez(
            os.path.join(output_dir, "time{}.npz".format(self.fold_idx)),
            start=start, end=time.time()
        )
        print(f"alpha: {alpha}")
        return os.path.join(output_dir, "params_fold{}.npz".format(self.fold_idx))
