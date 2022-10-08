import tensorflow.compat.v1 as tf

from deepsleepLite.nn import *

from deepsleepLite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SEQ_OF_EPOCHS,
                                        SAMPLING_RATE)

class SleepNetLite(object):

    def __init__(
            self,
            batch_size,
            input_dims,
            seq_length,
            n_classes,
            is_train,
            reuse_params,
            use_MC_dropout,
            freeze,
            name="sleepnetlite"
    ):
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.seq_length = seq_length
        self.n_classes = n_classes
        self.is_train = is_train
        self.reuse_params = reuse_params
        self.use_MC_dropout = use_MC_dropout
        self.freeze = freeze
        self.name = name

        self.activations = []
        self.layer_idx = 1
        self.monitor_vars = []

    def _build_placeholder(self):
     
        name = "x_train" if self.is_train else "x_valid"
        self.input_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=(self.batch_size, int(self.input_dims*self.seq_length), 1, 1),
            name= name + "_inputs"
        )

        # Target
        self.target_var = tf.compat.v1.placeholder(
            tf.int32,
            shape=[self.batch_size, ],
            name=name + "_targets"
        )

        # Target conditioned prob
        self.target_var_conditioned = tf.compat.v1.placeholder(
            tf.float32,
            shape=[self.batch_size, 5],
            name=name + "_targets_conditioned"
        )
        # Label smoothing alfa parameter
        self.alfa = tf.compat.v1.placeholder(
            tf.float32,
            shape=None,
            name=name + "_alfa"
        )

        # Conditional distributions on sleep stages
        self.conditional_distribution = tf.compat.v1.placeholder(
            tf.float32,
            shape=[self.batch_size, self.n_classes],
            name=name + "_smoothing_distribution"
        )

        # Conditional training
        self.id_cluster = tf.compat.v1.placeholder(
            tf.int32,
            shape=[self.batch_size, ],
            name=name + "_condition_encoding"
        )

    def _conv1d_layer(self, input_var, filter_size, n_filters, stride, wd=0, freeze=None):
        input_shape = input_var.get_shape()
        n_batches = input_shape[0]
        input_dims = input_shape[1]
        n_in_filters = input_shape[3]
        name = f"l{self.layer_idx}_conv"
        with tf.compat.v1.variable_scope(name) as scope:
            output, weights = conv_1d(name="conv1d", input_var=input_var,
                                      filter_shape=[filter_size, 1, n_in_filters, n_filters], stride=stride, bias=None,
                                      wd=wd, freeze_layer=freeze)

            output = batch_norm(name="bn", input_var=output, is_train=self.is_train, freeze_layer=freeze)

            output = tf.nn.relu(output, name="relu")

        self.activations.append((name, output))
        self.layer_idx += 1
        return output

    def build_model(self, input_var):
        # List to store the output of each CNNs
        output_conns = []

        ######### CNNs with small filter size at the first layer #########

        # Convolution
        network = self._conv1d_layer(input_var=input_var, filter_size=round(SAMPLING_RATE / 2), n_filters=64,
                                     stride=round(SAMPLING_RATE / 16), wd=1e-3, freeze=self.freeze)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=8, stride=8)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train and not self.freeze:
            network = tf.nn.dropout(network, rate = 1 - 0.5, name=name)
        elif self.use_MC_dropout:
            network = tf.nn.dropout(network, rate=1 - 0.5, name=name)
        else:
            network = tf.nn.dropout(network, rate = 0, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1, freeze=self.freeze)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1, freeze=self.freeze)
        network = self._conv1d_layer(input_var=network, filter_size=8, n_filters=128, stride=1, freeze=self.freeze)

        # Max pooling
        name = f"l{self.layer_idx}_pool"
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### CNNs with large filter size at the first layer #########

        # Convolution
        # network = self._conv1d_layer(input_var=input_var, filter_size=1024, n_filters=64, stride=128)
        network = self._conv1d_layer(input_var=input_var, filter_size=round(SAMPLING_RATE * 4), n_filters=64,
                                     stride=round(SAMPLING_RATE / 2), wd=1e-3, freeze=self.freeze)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=4, stride=4)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train and not self.freeze:
            network = tf.nn.dropout(network, rate = 1 - 0.5, name=name)
        elif self.use_MC_dropout:
            network = tf.nn.dropout(network, rate=1 - 0.5, name=name)
        else:
            network = tf.nn.dropout(network, rate = 0, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Convolution
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1, freeze=self.freeze)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1, freeze=self.freeze)
        network = self._conv1d_layer(input_var=network, filter_size=6, n_filters=128, stride=1, freeze=self.freeze)

        # Max pooling
        name = "l{}_pool".format(self.layer_idx)
        network = max_pool_1d(name=name, input_var=network, pool_size=2, stride=2)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Flatten
        name = "l{}_flat".format(self.layer_idx)
        network = flatten(name=name, input_var=network)
        self.activations.append((name, network))
        self.layer_idx += 1

        output_conns.append(network)

        ######### Aggregate and link two CNNs #########

        # Concat
        name = "l{}_concat".format(self.layer_idx)
        network = tf.concat(output_conns, 1, name=name)  ##CHANGE new version tensorflow  (1, output_conns, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        # Dropout
        name = "l{}_dropout".format(self.layer_idx)
        if self.is_train and not self.freeze:
            network = tf.nn.dropout(network, rate=1 - 0.5, name=name)
        elif self.use_MC_dropout:
            network = tf.nn.dropout(network, rate=1 - 0.5, name=name)
        else:
            network = tf.nn.dropout(network, rate=0, name=name)
        self.activations.append((name, network))
        self.layer_idx += 1

        return network

    def init_ops(self):
        self._build_placeholder()

        # Get loss and prediction operations
        with tf.compat.v1.variable_scope(self.name) as scope:

            # Reuse variables for validation
            if self.reuse_params:
                scope.reuse_variables()

            # Build model
            network = self.build_model(input_var=self.input_var)

            # Softmax linear
            name = f"l{self.layer_idx}_softmax_linear"
            network = fc(name=name, input_var=network, n_hiddens=self.n_classes, bias=0.0, wd=0)
            self.activations.append((name, network))
            self.layer_idx += 1

            # Outputs of softmax linear are logits
            self.logits = network

            # Uniform Label smoothing
            #name = "l{}_label_smoothing_uniform".format(self.layer_idx)
            #self.target_var_smoothed = label_smoothing_uniform(name=name, target_var=self.target_var, alfa=self.alfa, K=self.n_classes)

            # Soft Consensus Label smoothing
            name = "l{}_label_smoothing_soft_consensus".format(self.layer_idx)
            self.target_var_smoothed = label_smoothing_soft_consensus(name=name, target_var=self.target_var, target_var_conditioned=self.target_var_conditioned, alfa=self.alfa)



            ######### Compute loss #########

            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.target_var_smoothed,
                logits=self.logits,
                name="softmax_cross_entropy_with_logits"

            )  

            loss = tf.reduce_mean(loss, name="cross_entropy")

            # Regularization loss
            regular_loss = tf.add_n(
                tf.compat.v1.get_collection("losses", scope=scope.name + "\/"),
                name="regular_loss"
            )

            # Total loss
            self.loss_op = tf.add(loss, regular_loss)

            # Predictions
            self.pred_op = tf.argmax(self.logits, 1)
