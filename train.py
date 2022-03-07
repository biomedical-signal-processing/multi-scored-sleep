#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import tensorflow.compat.v1 as tf
from deepsleepLite.trainer import SleepNetLiteTrainer
from deepsleepLite.sleep_stages import (NUM_CLASSES,
                                        EPOCH_SEC_LEN,
                                        SEQ_OF_EPOCHS,
                                        SAMPLING_RATE)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', 'C:/Users/pedro/OneDrive - Politecnico di Torino/Desktop/DSN-L/pre_processing/DOD-O/',
                           """Directory where to load training data.""")
tf.app.flags.DEFINE_string('output_dir', '/output',
                           """Directory where to save trained models """
                           """and outputs.""")
tf.app.flags.DEFINE_integer('n_folds', 10,
                            """Number of cross-validation folds.""")
tf.app.flags.DEFINE_integer('fold_idx', 0,
                            """Index of cross-validation fold to train.""")
tf.app.flags.DEFINE_integer('pretrain_epochs', 150,
                            """Number of epochs for pretraining DeepFeatureNet.""")
tf.app.flags.DEFINE_boolean('resume', False,
                            """Whether to resume the training process.""")
tf.app.flags.DEFINE_boolean('freeze', False,
                            """Whether to freeze part of the network to fine-tune/transfer the knowledge on new data.""")
tf.app.flags.DEFINE_float('alpha',0,
                             "Hyperparameter Label Smoothing")


def pretrain(n_epochs):
    print(float(FLAGS.alpha))
    trainer = SleepNetLiteTrainer(
        data_dir=FLAGS.data_dir,
        output_dir=FLAGS.output_dir,
        n_folds=FLAGS.n_folds,
        fold_idx=FLAGS.fold_idx,  # indice del fold corrente
        batch_size=100,  # suddivide in 100 parti l'epoca e la da in pasto alla rete 1 pezzo per volta
        input_dims=EPOCH_SEC_LEN * SAMPLING_RATE,
        seq_length=SEQ_OF_EPOCHS,
        n_classes=NUM_CLASSES,
        interval_print_cm=5
    )
    pretrained_model_path = trainer.train(
        n_epochs=n_epochs,
        resume=FLAGS.resume,
        freeze=FLAGS.freeze,
        alpha=FLAGS.alpha
    )
    return pretrained_model_path


def main(argv=None):
    l = [0,1,2,3,4,5,6,7,8,9]
    #FLAGS.n_folds=len(l)
    #l = [0, 1, 2, 3, 4,]
    #l = list(range(0, 25))
     #l = [9]
    for fold_idx in l:
        FLAGS.fold_idx = fold_idx
        # Output dir
        output_dir = os.path.join(FLAGS.output_dir, f"fold{FLAGS.fold_idx}")
        if not FLAGS.resume:  # entra se FLAGS.resume == False
            if tf.io.gfile.exists(output_dir):  # se esiste
                tf.io.gfile.rmtree(output_dir)  # elimina tutto ciò che c'è in quel path
            tf.io.gfile.makedirs(output_dir)  # se non esite crea la cartella

        # FeatureNet
        pretrained_model_path = pretrain(
            n_epochs=FLAGS.pretrain_epochs
        )


if __name__ == "__main__":
    # Run Network
    tf.compat.v1.app.run()
