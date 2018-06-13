import os
import sys
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import tensorflow as tf
import numpy as np
import json

from cnn_model import CNN_Model
from load_data import *

from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

def get_weights(trainable_ops):
    o = {}
    for op in trainable_ops:
        if('kernel' in op.name):
            name = op.name.split("/")[0]
            o[name] = op
    return o

def get_activations(vis_ops):
    o = {}
    for op in vis_ops:
        if ('Relu' in op.name or 'maxpool' in op.name or 'out' in op.name ):
            name = op.name.split("/")[0]
            o[name] = op
    return o

def get_json_array(vis_ops, weight_ops, weight_values, activation_values, input_values, output_value, class_names):
    o = []
    input_shape = input_values.shape
    input_values = input_values.astype(np.float32)
    input_values = np.divide(input_values, 255)

    o.append((class_names[output_value], 'input', input_shape, np.zeros(input_shape).tolist(), input_shape, input_values.tolist())) #TODO: activations are dummies for now

    vis_op_names = set()

    for op in vis_ops:
        name = op.name.split("/")[0]
        if(name in vis_op_names):
            continue
        vis_op_names.add(name)
        if('maxpool' in op.name):
            act_val = activation_values[name]
            o.append((class_names[output_value], name, op.shape[1:3].as_list(), np.zeros(op.shape[1:3].as_list()).tolist(), act_val.shape, act_val.tolist()))

        elif(name in weight_ops.keys()):
            act_val = activation_values[name]
            shape = weight_ops[name].shape.as_list()
            weights = weight_values[name]
            o.append((class_names[output_value], name, shape, weights.tolist(), act_val.shape, act_val.tolist()))


    o.append(class_names)
    return o



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
        parser.add_argument("-ts", "--test_sample", type=int, dest="test_sample", action="store", default=0)

        args = parser.parse_args()

    except Exception as e:
        raise (e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

    for ts in range(args.test_sample, args.test_sample+1):
        test_sample = 2
        irace = args.irace

        model = CNN_Model("cnn_model",
                          cnn_group_num=args.cnn_group_num,
                          cnn_group_size=args.cnn_group_size,
                          cnn_base_filters=int(args.cnn_base_filters),
                          dense_layer_num=args.dense_layer_num,
                          dense_layer_units=int(args.dense_layer_units),
                          dropout=args.dropout
                          )

        args.dataset = ''
        x, y, x_test, y_test, width, class_names = load_dataset(args.dataset)

        dummy_x = np.expand_dims(x_test[test_sample], 0)
        dummy_y = np.expand_dims(y_test[test_sample], 0)

        for i in range(11):
            with tf.Session() as sess:
                new_saver = tf.train.import_meta_graph('ckpt/model.ckpt-{}.meta'.format(i))
                new_saver.restore(sess, 'ckpt/model.ckpt-{}'.format(i))

                print(sess.run(tf.report_uninitialized_variables()))

                input_op = sess.graph.get_collection("input_op")[0]

                vis_ops = tf.get_collection('VisOps')
                train_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                weight_ops = get_weights(train_ops)
                weight_values = sess.run(weight_ops)
                activation_values, input_values = sess.run([get_activations(vis_ops), input_op], {input_op: dummy_x})

                #print(activation_values['conv0'].tolist())

                json_base = get_json_array(vis_ops, weight_ops, weight_values, activation_values, input_values, dummy_y[0], class_names)

                outdir = "jsons/{}/".format(test_sample)
                filename = "{}epoch{}.json".format(outdir, i)
                if(not os.path.exists(os.path.split(filename)[0])):
                    os.makedirs(outdir)
                with open(filename, "w") as outfile:
                    try:
                        json.dump(json_base, outfile)
                    except Exception as e:
                        print(e.message)


if __name__ == "__main__":
    main(sys.argv)