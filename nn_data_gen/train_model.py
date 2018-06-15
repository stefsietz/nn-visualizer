import csv
import os
import sys
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import numpy as np
import tensorflow as tf

from cnn_model import CNN_Model
from load_data import *


def csv_to_batch(filename):
    with open(filename, newline='') as csvfile:
        outmat = np.zeros((10000, 1025), dtype=np.uint8)

        csvreader = csv.reader(csvfile, delimiter=',')
        r = 0
        for row in csvreader:
            if (r == 0):
                r+=1
                continue
            else:
                outmat[r - 1, 0] = int(row[0])
                for i in range(1, 1025):
                    outmat[r - 1, i] = int(row[i])
                r+=1

        np.save(os.path.splitext(filename)[0]+'.npy', outmat)
        return outmat

def main(argv=None):

    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)


    program_name = os.path.basename(sys.argv[0])
    program_usage = '''View or solve book embedding
USAGE
'''
    try:
        # Setup argument parser
        parser = ArgumentParser(description=program_usage, formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('input', metavar='input-file', type=str, nargs=1)
        parser.add_argument('output', metavar='output-file', type=str, nargs='?')
        parser.add_argument("-ir", "--irace", type=bool, dest="irace", action="store", default=False)
        parser.add_argument("-s", "--seed", type=int, dest="seed", action="store", default=50)
        parser.add_argument("-e", "--epochs", type=int, dest="epochs", action="store", default=50)
        parser.add_argument("-lr", "--learning_rate", type=float, dest="learning_rate", action="store", default=0.0001)
        parser.add_argument("-bs", "--batch_size", type=int, dest="batch_size", action="store", default="256")
        parser.add_argument("-cgs", "--cnn_group_size", type=int, dest="cnn_group_size", action="store", default=2)
        parser.add_argument("-cgn", "--cnn_group_num", type=int, dest="cnn_group_num", action="store", default=2)
        parser.add_argument("-cbf", "--cnn_base_filters", dest="cnn_base_filters", action="store", default="64")
        parser.add_argument("-dln", "--dense_layer_num", type=int, dest="dense_layer_num", action="store", default=1)
        parser.add_argument("-dlu", "--dense_layer_units", type=int, dest="dense_layer_units", action="store", default="256")
        parser.add_argument("-do", "--dropout", type=float, dest="dropout", action="store", default=0.2)
        parser.add_argument("-ds", "--dataset", type=str, dest="dataset", action="store", default='cifar10')

        args = parser.parse_args()

    except Exception as e:
        raise (e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

    irace = args.irace

    model = CNN_Model("cnn_model",
                      cnn_group_num=args.cnn_group_num,
                      cnn_group_size=args.cnn_group_size,
                      cnn_base_filters=int(args.cnn_base_filters),
                      dense_layer_num=args.dense_layer_num,
                      dense_layer_units=int(args.dense_layer_units),
                      dropout=args.dropout
                      )
    x, y, x_test, y_test, width, _ = load_dataset(args.dataset)

    x_inference = tf.placeholder(tf.uint8, [None, width, width, 1], name="X")
    y_inference = tf.placeholder(tf.uint8, [None], name="Y")

    EPOCHS = args.epochs
    BATCH_SIZE = int(args.batch_size)

    batch_size = BATCH_SIZE

    dataset_train = tf.data.Dataset.from_tensor_slices((x,y)).repeat().shuffle(buffer_size=500).batch(batch_size)
    dataset_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).repeat().shuffle(buffer_size=500).batch(batch_size)

    n_batches = x.shape[0] // BATCH_SIZE

    iter = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)

    features, labels = iter.get_next()

    logits = model.inference(features)
    test_logits = model.inference(features, False)

    loss = model.loss(logits, labels)
    loss_test = model.loss(test_logits, labels)

    train_accuracy = model.accuracy(logits, labels)
    test_accuracy = model.accuracy(test_logits, labels)

    loss_var = tf.Variable(0.0)
    acc_var = tf.Variable(0.0)

    s1 = tf.summary.scalar("loss", loss_var)
    s2 = tf.summary.scalar("accuracy", acc_var)

    summary = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

    train_init_op = iter.make_initializer(dataset_train)
    test_init_op = iter.make_initializer(dataset_test, name="test_init_op")

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('summaries' + '/train')
        test_writer = tf.summary.FileWriter('summaries' + '/test')

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=None)
        saver.save(sess, "ckpt/model.ckpt", global_step=0)
        for i in range(EPOCHS):
            sess.run(train_init_op)
            train_loss = 0
            train_acc = 0
            for _ in range(n_batches):
                __, loss_value, tr_acc = sess.run([train_op, loss, train_accuracy])

                train_loss += loss_value
                train_acc += tr_acc

            sess.run(test_init_op)
            test_loss, test_acc = sess.run([loss_test, test_accuracy])

            train_writer.add_summary(sess.run(summary, {loss_var: train_loss/n_batches, acc_var: train_acc/n_batches}), i)
            test_writer.add_summary(sess.run(summary, {loss_var: test_loss, acc_var: test_acc}), i)

            saver.save(sess, "ckpt/model.ckpt", global_step=i+1)

            if irace:
                pass
            else:
                print("Epoch: {}, Train loss: {:.4f}, Test loss: {:.4f}, Train accuracy: {:.4f}, Test accuracy: {:.4f} "\
                  .format(i,train_loss/n_batches, test_loss, train_acc/n_batches, test_acc))

        print(test_loss)

if __name__ == "__main__":
    main(sys.argv)




