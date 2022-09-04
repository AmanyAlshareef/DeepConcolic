#!/usr/bin/env python3
#To get the summary of a model architecture: python3 -m deepconcolic.eval_classifier --dataset mnist --model saved_models/mnist_complicated.h5

import yaml
import seaborn as sns
from scipy.special import rel_entr 
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, confusion_matrix, classification_report
import tensorflow as tf
from pomegranate import *
from pomegranate.distributions import ConditionalProbabilityTable, DiscreteDistribution
from art.estimators.classification import ClassifierMixin
from art.estimators.classification.keras import KerasClassifier
from art.metrics.metrics import empirical_robustness, clever_u
from deepconcolic.utils import lazy_activations_on_indexed_data
from utils import *
from utils import dataset_dict, predictions
from utils_funcs import rng_seed
from utils_args import *
from utils_io import OutputDir
from dbnc import BNAbstraction, layer_setup, interval_repr, KBinsFeatureDiscretizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool
from engine import *

from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from numpy import array
import datasets
import plugins
import scipy
import copy
import math
from math import log2, log, e
 

# ---

def load_model (filename, print_summary = True):
  tf.compat.v1.disable_eager_execution ()
  dnn = keras.models.load_model (filename)
  if print_summary:
    dnn.summary ()
  return dnn

def load_dataset (name):
  train, test, _, _, _ = datasets.load_by_name (name)
  return raw_datat (*train, name), raw_datat (*test, name)

def fit_data (dnn, bn_abstr, data, indexes):
  indexes = np.arange (len (data.data)) if indexes is None else indexes
  np1 (f'| Fitting BN with {len (indexes)} samples... ')
  lazy_activations_on_indexed_data \
    (bn_abstr.fit_activations, dnn, data, indexes,
     layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ],
     pass_kwds = False)
  c1 ('done')

def fit_data_sample (dnn, bn_abstr, data, size, rng):
  bn_abstr.reset_bn ()
  idxs = np.arange (len (data.data))
  if size is not None:
    idxs = rng.choice (a = idxs, axis = 0, size = min (size, len (idxs)))
  fit_data (dnn, bn_abstr, data, idxs)


def List_flatten(t):
    return [item for sublist in t for item in sublist]


def BN_prediction(test_object, bn_abstr, adv_data = None):  
  layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers]
  indexes = np.arange (len (test_object.raw_data.data))
  if adv_data is not None:
    test_object.raw_data.data = adv_data
  obs_vals = lazy_activations_on_indexed_data \
    (bn_abstr.dimred_n_discretize_activations, test_object.dnn,
     test_object.raw_data, indexes,
     layer_indexes = layer_indexes,
     pass_kwds = False)

  true_labels = test_object.raw_data.labels[indexes]

  #To delet
  true_labels = true_labels.reshape(len(test_object.raw_data.labels), 1) 
  final_data = np.concatenate([obs_vals, true_labels], axis = 1)
  none_data = final_data.copy().astype(object)

  none_data[:, -1] = None
 
  predictions = bn_abstr.N.predict(none_data)
  predictions = np.array(predictions)
  
  true_labels = final_data[:, -1]
  pred_labels = predictions[:, -1]
  #print("true_labels: {} \npred_labels: {} \n".format(true_labels[:15], pred_labels[:15]))

  true_labels = true_labels.astype(int)
  pred_labels = pred_labels.astype(int)
  BN_accuracy = 100.0 * np.sum(np.equal(true_labels, pred_labels)) / len(true_labels)
  
  return BN_accuracy
# ---

#Image Transformation
def imgen (test_object):
  datagen = ImageDataGenerator (
    rotation_range = 15, # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.2,   # Randomly zoom image 
    shear_range = 0.2, # shear angle in counter-clockwise direction in degrees  
    width_shift_range=0.08, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.08, # randomly shift images vertically (fraction of total height)
    vertical_flip=True,      # randomly flip images
  )
  np1 ('| Fitting image data generator... ')
  datagen.fit (test_object.train_data.data)
  c1 ('done')
  X, Y = [], []
  test_size = len (test_object.raw_data.data)
  np1 (f'| Generating {test_size} new images... ')
  for x, y in datagen.flow (test_object.raw_data.data[:test_size],
                            test_object.raw_data.labels[:test_size]
                            # ,batch_size=1, save_to_dir="outs/NewGenData"
                            # ,save_prefix="image", save_format="jpg"
                            ):
    X.extend (x)
    Y.extend (y)
    if len (X) >= test_size:
      break
  c1 ('done')
  return raw (np.array (X), np.array (Y))

# ---

def _clr (dnn, art_cls_params = None):
  art_cls_params = some (art_cls_params, dict (clip_values = (0, 1),
                                                 use_logits = False))
  try:
    from tensorflow import keras as K
    if isinstance (dnn, K.Model):
       dnn = KerasClassifier (dnn, **art_cls_params)
  except:
    pass

  if not isinstance (dnn, ClassifierMixin):
    raise ValueError (f'unsupported model of type {type (dnn)}')

  return dnn

# ---

parser = argparse.ArgumentParser (description = 'BN abstraction manager')
parser.add_argument ('--dataset', dest='dataset', required = True,
                     help = "selected dataset", choices = datasets.choices)
parser.add_argument ('--model', dest='model', required = True,
                     help = 'neural network model (.h5)')
parser.add_argument ('--rng-seed', dest="rng_seed", metavar="SEED", type=int,
                     help="Integer seed for initializing the internal random number "
                    "generator, and therefore get some(what) reproducible results")
subparsers = parser.add_subparsers (title = 'sub-commands', required = True,
                                    dest = 'cmd')

# ---

ap_create = subparsers.add_parser ('create')
add_abstraction_arg (ap_create)
ap_create.add_argument ("--outputs", dest="outputs", required=True,
                        help="the output test data directory", metavar="DIR")
ap_create.add_argument ("--layers", dest = "layers", nargs = "+", metavar = "LAYER",
                        help = 'considered layers (given by name or index)')
ap_create.add_argument ('--train-size', '-ts', type = int,
                        help = 'train dataset size (default is all)',
                        metavar = 'INT')
ap_create.add_argument ('--feature-extraction', '-fe',
                        choices = ('pca', 'ipca', 'ica',), default = 'pca',
                        help = 'feature extraction technique (default is pca)')
ap_create.add_argument ('--num-features', '-nf', type = int, default = 2,
                        help = 'number of extracted features for each layer '
                        '(default is 2)', metavar = 'INT')
ap_create.add_argument ('--num-intervals', '-ni', type = int, default = 2,
                        help = 'number of intervals for each extracted feature '
                        '(default is 2)', metavar = 'INT')
ap_create.add_argument ('--discr-strategy', '-ds',
                        choices = ('uniform', 'quantile',), default = 'uniform',
                        help = 'discretisation strategy (default is uniform)')
ap_create.add_argument ('--extended-discr', '-xd', action = 'store_true',
                        help = 'use extended partitions')

# python3 -m deepconcolic.dbnabstr_infer --dataset mnist --model saved_models/mnist_complicated.h5 create outs/bnInfer.pkl --feature-extraction pca --num-features 3 --num-intervals 5 --layers max_pooling2d_1 max_pooling2d_2 activation_5 --outputs outs
# python3 -m deepconcolic.dbnabstr_infer --dataset cifar10 --model saved_models/cifar10_complicated.h5 create outs/bnInfer.pkl --feature-extraction pca --num-features 3 --num-intervals 5 --layers max_pooling2d_1 max_pooling2d_2 activation_5 --outputs outs

def create (test_object,
            dataset =  None,
            outputs = None,
            layers = None,
            train_size = None,
            feature_extraction = None,
            num_features = None,
            num_intervals = None,
            discr_strategy = None,
            extended_discr = False,
            abstraction = None,
            **_):
  if layers is not None:
    test_object.set_layer_indices (int (l) if l.isdigit () else l for l in layers)
  n_bins = num_intervals - 2 if extended_discr else num_intervals
  if n_bins < 1:
    raise ValueError (f'The total number of intervals for each extracted feature '
                      f'must be strictly positive (got {n_bins} '
                      f'with{"" if extended_discr else "out"} extended discretization)')
  feats = dict (decomp = feature_extraction, n_components = num_features)
  discr = dict (strategy = discr_strategy, n_bins = n_bins, extended = extended_discr)
  setup_layer = lambda l, i, **kwds: \
    layer_setup (l, i, feats, discr, discr_n_jobs = 8)
  clayers = get_cover_layers \
    (test_object.dnn, setup_layer, layer_indices = test_object.layer_indices,
     activation_of_conv_or_dense_only = False,
     exclude_direct_input_succ = False,
     exclude_output_layer = False)
  bn_abstr = BNAbstraction (clayers, dump_abstraction = False, outdir = OutputDir (outputs, log = True))
  lazy_activations_on_indexed_data \
    (bn_abstr.initialize, test_object.dnn, test_object.train_data,
     np.arange (min (train_size or sys.maxsize, len (test_object.train_data.data))),
     [fl.layer_index for fl in clayers],
     fit_with_training_data = True)
  #bn_abstr.dump_abstraction (pathname = abstraction_path (abstraction))
  #bn_abstr.dump_bn ('bn4trained', 'training dataset') 
  

  BNlast_layer_nodes = []
  for i, state in enumerate(bn_abstr.N.states): 
    if i in range(num_features*len(layers)-num_features, num_features*len(layers)):
      BNlast_layer_nodes.append(state.name)

  BN_distributions = []
  for state in bn_abstr.N.states:
    #if state.name in BNlast_layer_nodes:  
    BN_distributions.append(state.distribution)

  # Transform the new input taken from test dataset to evidences/observations
  rng = np.random.default_rng (randint ())
  indexes = np.arange (len (test_object.train_data.data))
  layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ]
  sample_data = lazy_activations_on_indexed_data \
    (bn_abstr.dimred_n_discretize_activations, test_object.dnn,
     test_object.train_data, indexes,
     layer_indexes = layer_indexes,
     pass_kwds = False)

  true_labels = test_object.train_data.labels[indexes]
  true_labels = true_labels.reshape(len(test_object.train_data.labels), 1) 

  final_data = np.concatenate([sample_data, true_labels], axis = 1)
  #np.save("final_data_s.npy", final_data)

  #Calculate the prediction node's CPT and add it into the BN 
  label_dist = ConditionalProbabilityTable.from_samples(final_data, BN_distributions)
  label_node = Node(label_dist, name = "prediction")
  bn_abstr.N.add_node(label_node)
  #print(label_dist.to_dict()["table"])
  
  for state in bn_abstr.N.states:
    if state.name not in ["prediction"]:  #in BNlast_layer_nodes:
      bn_abstr.N.add_edge(state, label_node)

  np1('| Fitting BN with prediction node data... ')
  bn_abstr.N.bake()
  bn_abstr.N.fit(final_data, n_jobs=5) 
  c1('done')
  # bn_abstr.N.plot()
  # plt.show()
  #bn_abstr.dump_bn ('bn4prediction', 'extra node data')
  
  # Perform inference/prediction
  x_test, y_test = test_object.raw_data.data, test_object.raw_data.labels 
  from keras.utils import to_categorical
  y_test = to_categorical(y_test)

  print(f'| Evaluating the DNN and BN prediction accuracy with {(num_features*len(layers))} BN nodes and {num_intervals} intervals each, on {dataset} dataset ({len (x_test)} samples): ')
  # Evaluate the DNN classifier on the test set
  loss, acc = test_object.dnn.evaluate(x_test, y_test, verbose=1)
  NN_accuracy = 100.0 * acc
  # Evaluate the BN classifier on the test set
  BN_accuracy = BN_prediction(test_object, bn_abstr)
  
  print("  NN prediction accuracy on raw test samples: {:.2f}%".format(NN_accuracy))
  print("  BN prediction accuracy on raw test samples: {}%".format(BN_accuracy))
  #print("| BN confusion_matrix: {}".format(confusion_matrix(true_test_labels, pred_labels)))

  # Craft adversarial samples with FGSM
  model = _clr (test_object.dnn)
  epsilon = 0.1  # Maximum perturbation
  pgdargs = dict (batch_size = 128)
  attacks = dict (
      fgsm = FastGradientMethod(model, eps=epsilon),
      #pgdlinf = ProjectedGradientDescent (model, norm = 'np.inf',
      #                                           eps = .1, eps_step = .01, **pgdargs),
      pgdl2 = ProjectedGradientDescent (model, norm = 2,
                                               eps = 10, eps_step = 1, **pgdargs),
      deepfool = DeepFool (model, batch_size = 128)
      )
 
  # Evaluate the NN and BN on the adversarial examples
  for a in attacks:
    attack = attacks[a]
    x_adv = attack.generate (x = x_test)
  
    NN_preds = np.argmax (model.predict (x_adv), axis = 1)
    NN_acc_adv = np.sum (NN_preds == test_object.raw_data.labels) / len (test_object.raw_data.data)
    BN_acc_adv = BN_prediction(test_object, bn_abstr, x_adv)
    print("  NN prediction accuracy on %s adversarial samples: %.2f%% " % (a, NN_acc_adv * 100))
    print("  BN prediction accuracy on {} adversarial samples: {}%".format(a, BN_acc_adv))


  # Transformed data
  # test_object.raw_data = org_raw_data
  # test_object.raw_data = imgen (test_object)

  # Xt_test, Yt_test = test_object.raw_data.data, test_object.raw_data.labels
  # Yt_test = to_categorical(Yt_test)
 
  # loss, acc_t = test_object.dnn.evaluate(Xt_test, Yt_test, verbose=1)
  # NN_accuracy_t = 100.0 * acc_t
  # BN_accuracy_t = BN_prediction(test_object, bn_abstr)
  # print("  NN prediction accuracy on transformed data: {:.2f}%".format(NN_accuracy_t))
  # print("  BN prediction accuracy on transformed data: {}%".format(BN_accuracy_t))


ap_create.set_defaults (cmd = create)

# ---


def draw_vector(v0, v1, ax=None):
  ax = ax or plt.gca()
  arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
  ax.annotate('', v1, v0, arrowprops=arrowprops)


def get_args (args = None, parser = parser):
  args = parser.parse_args () if args is None else args
  # Initialize with random seed first, if given:
  try: rng_seed (args.rng_seed)
  except ValueError as e:
    sys.exit (f'Invalid argument given for \`--rng-seed\': {e}')
  return args


def main (args = None, parser = parser, pp_args = (pp_abstraction_arg (),)):
  try:
    args = get_args (args, parser = parser)
    # args = reduce (lambda args, pp: pp (args), pp_args, args)
    for pp in pp_args: pp (args)
    test_object = test_objectt (load_model (args.model),
                                *load_dataset (args.dataset))

    if 'cmd' in args:
      args.cmd (test_object, **vars (args))
    else:
      parser.print_help ()
      sys.exit (1)
  except ValueError as e:
    sys.exit (f'Error: {e}')
  except FileNotFoundError as e:
    sys.exit (f'Error: {e}')
  except KeyboardInterrupt:
    sys.exit ('Interrupted.')

# ---

if __name__=="__main__":
  main ()

# ---

