import os
import sys
import re
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

import tensorflow as tf
import numpy as np
import json

from cnn_model import CNN_Model
from load_data import *

from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

def get_weight_ops(trainable_ops):
    o = {}
    for op in trainable_ops:
        if('kernel' in op.name):
            name = op.name.split("/")[0]
            o[name] = op
    return o

def get_activation_ops(vis_ops):
    o = {}
    for op in vis_ops:
        if ('Relu' in op.name or 'maxpool' in op.name or 'out' in op.name ):
            name = op.name.split("/")[0]
            o[name] = op
    return o


def get_structure_json_list(vis_ops, weight_ops, input_op, class_names):
    out_list = []
    input_shape = input_op.shape.as_list()
    input_shape = [1 if x is None else x for x in input_shape]
    out_list.append(('input', input_shape))

    vis_op_names = set()

    for op in vis_ops:
        name = op.name.split("/")[0]
        if(name in vis_op_names):
            continue
        vis_op_names.add(name)
        if('maxpool' in op.name):
            shape = op.shape[1:3].as_list()
            out_list.append((name, shape))

        elif(name in weight_ops.keys()):
            shape = weight_ops[name].shape.as_list()
            out_list.append((name, shape))

    out_list.append(class_names)
    return out_list


def get_weight_json_list(vis_ops, weight_ops, weight_values):
    weight_list = []

    weight_list.append(('input', [1], [0]))

    vis_op_names = set()

    for op in vis_ops:
        name = op.name.split("/")[0]
        if(name in vis_op_names):
            continue
        vis_op_names.add(name)
        if('maxpool' in op.name or 'input' in op.name):
            shape = op.shape[1:3].as_list()
            weight_list.append((name, shape, np.zeros(shape).tolist()))

        elif(name in weight_ops.keys()):
            shape = weight_ops[name].shape.as_list()
            weights = weight_values[name]
            weight_list.append((name, shape, weights.tolist()))

    return weight_list

def get_activation_json_list(vis_ops, activation_ops, activation_values, input_values, groundtruths, class_names):
    activation_list = []
    input_shape = input_values.shape
    input_values = input_values.astype(np.float32)
    input_values = np.divide(input_values, 255)

    activation_list.append(('input', input_shape, input_values.tolist())) #TODO: activations are dummies for now

    vis_op_names = set()

    for op in vis_ops:
        name = op.name.split("/")[0]
        if(name in vis_op_names):
            continue

        vis_op_names.add(name)

        if(name in activation_ops.keys()):
            act_val = activation_values[name]
            activation_list.append((name, act_val.shape, act_val.tolist()))

        if('out' in  name):
            out_vals = activation_values[name]

            pred_vals = [np.argmax(x) for x in out_vals]
            predictions = [class_names[x] for x in pred_vals]

            groundtruth_classes = [class_names[x] for x in groundtruths]

    activation_list.append((groundtruth_classes, predictions))

    return activation_list

def get_checkpoint_files(dir):
    ckpt_list = os.listdir(dir)
    meta_list = [x for x in ckpt_list if '.meta' in x]
    meta_dict = {}
    for file in meta_list:
        regex = re.compile(r'\d+')
        file_number = regex.findall(file)
        meta_dict[int(file_number[-1])] = file

    indexlist = sorted(meta_dict.keys())
    return indexlist, meta_dict

def get_sample_inds(args):
    sample_range = (args.test_sample, args.test_sample+1)
    if(args.test_sample_range is not ''):
        sample_r = args.test_sample_range.split('-')
        sample_range = (int(sample_r[0]), int(sample_r[1])+1)

    sample_list = [x for x in range(sample_range[0], sample_range[1])]
    sample_inds = np.array(sample_list)

    return sample_inds

def main(argv=None):

    if argv is None:
        argv = sys.argv

    try:
        # Setup argument parser
        parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('input', metavar='input-file', type=str, nargs=1)
        parser.add_argument('output', metavar='output-file', type=str, nargs='?')
        parser.add_argument("-c", "--ckpt", type=str, dest="ckpt", action="store", default='ckpt')
        parser.add_argument("-j", "--json", type=str, dest="json", action="store", default='jsons')
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
        parser.add_argument("-tsr", "--test_sample_range", type=str, dest="test_sample_range", action="store", default='')

        args = parser.parse_args()

    except Exception as e:
        raise (e)
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

    sample_inds = get_sample_inds(args)

    irace = args.irace

    model = CNN_Model("cnn_model",
                      cnn_group_num=args.cnn_group_num,
                      cnn_group_size=args.cnn_group_size,
                      cnn_base_filters=int(args.cnn_base_filters),
                      dense_layer_num=args.dense_layer_num,
                      dense_layer_units=int(args.dense_layer_units),
                      dropout=args.dropout
                      )

    x, y, x_test, y_test, width, class_names = load_dataset(args.dataset)

    sample_x = x_test[sample_inds]
    sample_y = y_test[sample_inds]


    indexlist, meta_dict = get_checkpoint_files(args.ckpt)

    for i in indexlist:
        with tf.Session() as sess:
            metafile = os.path.join(args.ckpt, meta_dict[i])
            ckpt_file = os.path.splitext(metafile)[0]

            new_saver = tf.train.import_meta_graph(metafile)
            new_saver.restore(sess, ckpt_file)

            print(sess.run(tf.report_uninitialized_variables()))

            input_op = sess.graph.get_collection("input_op")[0]

            vis_ops = tf.get_collection('VisOps')
            trainable_ops = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            weight_ops = get_weight_ops(trainable_ops)
            weight_values = sess.run(weight_ops)

            activation_ops = get_activation_ops(vis_ops)

            activation_values, input_values = sess.run([activation_ops, input_op], {input_op: sample_x})

            structure_json_base = get_structure_json_list(vis_ops, weight_ops, input_op, class_names)
            weight_json_base = get_weight_json_list(vis_ops, weight_ops, weight_values)
            activation_json_base = get_activation_json_list(vis_ops, activation_ops, activation_values, input_values, sample_y, class_names)

            dataset_name = args.dataset

            outdir = args.json
            structure_filename = os.path.join(outdir, "model_structure.json")
            weight_filename = os.path.join(outdir, "model_weights_epoch{0:03d}.json".format(i))
            activation_filename = os.path.join(outdir, "model_activations_epoch{0:03d}.json".format(i))
            if(not os.path.exists(outdir)):
                os.makedirs(outdir)

            with open(structure_filename, "w") as outfile:
                try:
                    json.dump(structure_json_base, outfile)
                except Exception as e:
                    print(e.message)

            with open(weight_filename, "w") as outfile:
                try:
                    json.dump(weight_json_base, outfile)
                except Exception as e:
                    print(e.message)

            with open(activation_filename, "w") as outfile:
                try:
                    json.dump(activation_json_base, outfile)
                except Exception as e:
                    print(e.message)


if __name__ == "__main__":
    main(sys.argv)