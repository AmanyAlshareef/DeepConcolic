#!/usr/bin/env python3
import argparse
from . import datasets
from . import plugins
from pathlib import Path
from .utils_io import *
from .utils_funcs import *

def report (dnn, test_data):
  from utils import predictions
  from sklearn.metrics import classification_report

  X_test, Y_test = test_data.data, test_data.labels
  h1 ('Classificaion Report')
  tp1 (f'Evaluating DNN on {len (X_test)} test samples...')
  Y_dnn = predictions (dnn, X_test)
  return classification_report (Y_test, Y_dnn, **_)

available_metrics['accuracy'] = \
  dict (aliases = ('acc',),
        func = report,
        interactive_params = dict (print_header = True,
                                   zero_division = 0))
default_metrics += ['accuracy']

# ---

try:
  from art.estimators.classification import ClassifierMixin
  from art.estimators.classification.keras import KerasClassifier
  from art.metrics.metrics import empirical_robustness, clever_u

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

  # ER

  def report_on_empirical_robustness (dnn,
                                      data_samples,
                                      print_header = False,
                                      method = "fgsm",
                                      art_cls_params = None,
                                      **er_params):
    assert method in ('fgsm', 'hsj')
    dnn = _clr (dnn, art_cls_params = art_cls_params)

    X_test = data_samples.data[:1000]
    if print_header:
      h1 ('Empirical robustness report')
      tp1 (f'Evaluating on {len (X_test)} test samples...')
    # params = {"eps_step": 1.0, "eps": 1.0}
    # method, params = "fgsm", {"eps": 0.3, "eps_step": 0.1}
    emp_robust = empirical_robustness (dnn, X_test, method, er_params)
    return emp_robust

  available_metrics['empirical_robustness'] = \
    dict (aliases = ('emp-rob', 'robustness', 'ER'),
          func = report_on_empirical_robustness,
          # customizable_params = dict (method = ('fgsm', 'hsj')),
          interactive_params = dict (print_header = True,
                                     eps = np.array([0.01, 0.02, 0.03]),
                                     eps_step = np.array([0.001, 0.002, 0.003]),
                                     method = 'fgsm'))

  # CLEVER

  def report_on_clever (dnn,
                        data_samples,
                        print_header = False,
                        art_cls_params = None,
                        **cu_params):
    dnn = _clr (dnn, art_cls_params = art_cls_params)
    X_tests = data_samples.data if isinstance (data_samples, raw_datat) else data_samples

    if print_header:
      h1 ('CLEVER report')
    res = []
    for i, X_test in enumerate (X_tests[0 : min (len (X_tests), 3)]):
      if print_header:
        tp1 (f'Evaluating on test sample {i}...')
        # params = {"eps_step": 1.0, "eps": 1.0}
        # method, params = "fgsm", {"eps": 0.3, "eps_step": 0.1}
      res += [clever_u (dnn, X_test, **cu_params)]
    return res

  available_metrics['clever'] = \
    dict (aliases = ('CLEVER',),
          func = report_on_clever,
          interactive_params = dict (print_header = True,
                                     nb_batches = 50,
                                     batch_size = 256,
                                     radius = 1.,
                                     norm = np.inf))
except:
  pass

for m in tuple (available_metrics.keys ()):
  d = available_metrics[m]
  for a in d['aliases']:
    available_metrics[a] = d

parser = argparse.ArgumentParser \
  (description = 'Evaluate various metrics on classifier neural networks',
   prog = 'python3 -m deepconcolic.eval_classifier',
   prefix_chars = '-+',
   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument ('--model', dest = 'model_spec',
                     help = 'the input neural network model (.h5 or "vgg16")')
# parser.add_argument("--vgg16-model", dest='vgg16',
#                     help="use keras's default VGG16 model (ImageNet)",
#                     action="store_true")
# parser.add_argument("--inputs", dest="inputs", default="-1",
#                     help="the input test data directory", metavar="DIR")
# parser.add_argument("--rng-seed", dest="rng_seed", metavar="SEED", type=int,
#                     help="Integer seed for initializing the internal random number "
#                     "generator, and therefore get some(what) reproducible results")
parser.add_argument ("--dataset", dest = 'dataset_spec',
                     help = "selected dataset", choices = datasets.choices)
parser.add_argument ("--extra-tests", '+i', dest = 'extra_testset_dirs', metavar = "DIR",
                     type = Path, nargs = "+",
                     help = "additonal directories of test images")
parser.add_argument ('--metrics', '-m', nargs='+',
                     choices = available_metrics.keys (),
                     default = default_metrics)

def main (model_spec = None,
          dataset_spec = None,
          extra_testset_dirs = None,
          metrics = None):

  assert dataset_spec in datasets.choices
  from .utils import dataset_dict, load_model

  dd = dataset_dict (dataset_spec)
  test_data = dd['test_data']
  del dd

  if args.extra_testset_dirs is not None:
    for d in args.extra_testset_dirs:
      np1 (f'Loading extra image testset from `{str(d)}\'... ')
      x, y, _, _, _ = datasets.images_from_dir (str (d))
      X_test = np.concatenate ((X_test, x))
      y_test = np.concatenate ((Y_test, y))
      print ('done.')

  dnn = load_model (args.model)
  dnn.summary ()

  report (dnn, test_data)

if __name__=="__main__":
  try:
    main ()
  except KeyboardInterrupt:
    sys.exit('Interrupted.')
