#!/usr/bin/env python3
from .utils import *
from .utils_funcs import rng_seed, randint
from .utils_args import *
from .dbnc import DiscreteFAbstraction, BNAbstraction, GMMAbstraction
from .dbnc import interval_repr
from .dbnc import parse_dimred_specs, dimred_specs_doc
from .dbnc import flayer_setup, discr_layer_setup
from tabulate import tabulate
from . import plugins
from . import datasets
import scipy

# ---

def load_model (filename, print_summary = True,
                disable_eager_execution = True):
  if disable_eager_execution:
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

def eval_coverages (dnn, bn_abstr, data, size, rng):
  fit_data_sample (dnn, bn_abstr, data, size, rng)
  return dict (bfc = bn_abstr.bfc_coverage (),
               bfdc = bn_abstr.bfdc_coverage ())

def eval_probas (dnn, bn_abstr, data, size, rng, indexes = None):
  if indexes is None:
    indexes = np.arange (len (data.data))
    if size is not None:
      indexes = rng.choice (a = indexes, axis = 0, size = min (size, len (indexes)))
  probas = lazy_activations_on_indexed_data \
    (bn_abstr.activations_probas, dnn, data, indexes,
     layer_indexes = [ fl.layer_index for fl in bn_abstr.flayers ],
     pass_kwds = False)
  return dict (probas = probas,
               stats = scipy.stats.describe (probas))

# ---

def add_abstraction_param_args (ap, with_discretization = True):
  # gp = ap.add_mutually_exclusive_group (required = False)
  # basic = gp.add_argument_group ('basic specification')
  # basic.add_argument ('--feature-extraction', '-fe',
  #                     choices = ('pca', 'ipca', 'ica',), default = 'pca',
  #                     help = 'feature extraction technique (default is pca)')
  # basic.add_argument ('--num-features', '-nf', type = int, default = 2,
  #                     help = 'number of extracted features for each layer '
  #                     '(default is 2)', metavar = 'INT')
  ap.add_argument ('--dimred-specs', '-dr', type = str, default = None,
                   metavar = 'SPEC', help = dimred_specs_doc)
  if with_discretization:
    ap.add_argument ('--num-intervals', '-ni', type = int, default = 2,
                     help = 'number of intervals for each extracted feature '
                     '(default is 2)', metavar = 'INT')
    ap.add_argument ('--discr-strategy', '-ds',
                     choices = ('uniform', 'quantile',), default = 'uniform',
                     help = 'discretisation strategy (default is uniform)')
    ap.add_argument ('--extended-discr', '-xd', action = 'store_true',
                     help = 'use extended partitions')

def add_abstraction_args (ap, **_):
  ap.add_argument ("--layers", dest = "layers", nargs = "+", metavar = "LAYER",
                   help = 'considered layers (given by name or index)')
  ap.add_argument ('--train-size', '-ts', type = int,
                   help = 'train dataset size (default is all)',
                   metavar = 'INT')
  add_abstraction_param_args (ap, **_)

def init_flayers (test_object,
                  layers = None,
                  feature_extraction = None,
                  num_features = None,
                  dimred_specs = None,
                  num_intervals = None,
                  discr_strategy = None,
                  extended_discr = False,
                  **_):
  if layers is not None:
    test_object.set_layer_indices (int (l) if l.isdigit () else l for l in layers)
  if dimred_specs is not None:
    feats = parse_dimred_specs (test_object, dimred_specs)
  else:
    feats = dict (decomp = feature_extraction, n_components = num_features)
  if num_intervals is not None:
    n_bins = num_intervals - 2 if extended_discr else num_intervals
    if n_bins < 1:
      raise ValueError (f'The total number of intervals for each extracted feature '
                        f'must be strictly positive (got {n_bins} '
                        f'with{"" if extended_discr else "out"} extended discretization)')
    discr = dict (strategy = discr_strategy, n_bins = n_bins, extended = extended_discr)
    setup_layer = lambda *_, **__: discr_layer_setup (*_, feats, discr, discr_n_jobs = 8)
  else:
    setup_layer = lambda *_, **__: flayer_setup (*_, feats)
  clayers = get_cover_layers \
    (test_object.dnn, setup_layer, layer_indices = test_object.layer_indices,
     activation_of_conv_or_dense_only = False,
     exclude_direct_input_succ = False,
     exclude_output_layer = False)
  return clayers

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

ap = subparsers.add_parser ('create')
add_abstraction_arg (ap)
add_abstraction_args (ap)
ap.add_argument ('--abstraction-only', action = 'store_true')
ap.add_argument ('--jobs', '-j', type = int)

# TODO: learn structure after initialization.
def create (test_object,
            train_size = None,
            abstraction = None,
            abstraction_only = False,
            jobs = 1,
            dump_abstraction = True,
            return_abstraction = False,
            rng = None,
            **_):
  rng = rng or np.random.default_rng (randint ())
  clayers = init_flayers (test_object, **_)
  if abstraction_only:
    abstr = DiscreteFAbstraction (clayers)
  else:
    abstr = BNAbstraction (clayers, dump_abstraction = False,
                           bn_abstr_n_jobs = jobs)
  lazy_activations_on_indexed_data \
    (abstr.initialize, test_object.dnn, test_object.train_data,
     rng.choice (len (test_object.train_data.data),
                 min (train_size or sys.maxsize, len (test_object.train_data.data)),
                 replace = False),
     # np.arange (min (train_size or sys.maxsize, len (test_object.train_data.data))),
     [fl.layer_index for fl in clayers]# ,
     # fit_with_training_data = True
     )
  if dump_abstraction:
    abstr.dump_abstraction (pathname = abstraction_path (abstraction))
  # bn_abstr.dump_model ('model', 'training data', OutputDir ())
  if return_abstraction:
    return abstr

ap.set_defaults (cmd = create)

# ---

# ap = subparsers.add_parser ('gmm')
# add_abstraction_arg (ap)
# add_abstraction_args (ap, with_discretization = False)
# ap.add_argument ('--jobs', '-j', type = int)
# ap.add_argument ('--learn-model-train-ratio', type = float)

# def gmm (test_object,
#          train_size = None,
#          abstraction = None,
#          learn_model_train_ratio = 1.,
#          jobs = 1,
#          num_intervals = None,
#          **_):
#   clayers = init_flayers (test_object, **_)
#   abstr = GMMAbstraction (clayers, dump_abstraction = False,
#                           learn_model_train_ratio = learn_model_train_ratio,
#                           bn_abstr_n_jobs = jobs)
#   lazy_activations_on_indexed_data \
#     (abstr.initialize, test_object.dnn, test_object.train_data,
#      np.arange (min (train_size or sys.maxsize, len (test_object.train_data.data))),
#      [fl.layer_index for fl in clayers])
#   abstr.dump_abstraction (pathname = abstraction_path (abstraction))

# ap.set_defaults (cmd = gmm)

# ---

ap_show = subparsers.add_parser ('show')
add_abstraction_arg (ap_show)

def show (test_object,
          abstraction = None,
          **_):
  bn_abstr = BNAbstraction.from_file (abstraction_path (abstraction),
                                      dnn = test_object.dnn,
                                      log = False)
  table = [
    [str (fl)] +
    [ '\n'.join (str (f) for f in range (fl.num_features)) ] +
    [ '\n'.join (', '.join (interval_repr (i) for i in fi_intervals)
                 for fi_intervals in fl.intervals) ]
    for fl in bn_abstr.flayers
  ]
  h1 ('Extracted Features and Associated Intervals')
  p1 (tabulate (table, headers = ('Layer', 'Feature', 'Intervals')))

ap_show.set_defaults (cmd = show)

# ---

def random_shift_feature (x_ref, fi, fl, f, rng = None):
  x = x_ref.copy ()
  x[:,fi] = np.select ([x[:,fi] == 0, x[:,fi] == fl.num_feature_parts [f] - 1],
                       [           1,            fl.num_feature_parts [f] - 2],
                       x[:,fi] + 1 - 2 * rng.integers (1, size = len (x)))
  assert np.amax (np.abs (x_ref[:,fi] - x[:,fi])) == 1
  assert np.amin (np.abs (x_ref[:,fi] - x[:,fi])) == 1
  return x

# ---

ap = subparsers.add_parser ('perturb-features')
add_abstraction_arg (ap)
ap.add_argument ('--iters', '-N', type = int, default = 1)
ap.add_argument ('--jobs', '-j', type = int, default = 5)

def perturb_features_ (dnn, data, abstr,
                       from_discrete_abstraction = BNAbstraction.from_discrete_abstraction,
                       learn_model_structure = False,
                       learn_model_structure_train_ratio = .1,
                       return_dict = False,
                       idxs = slice (None),
                       iters = 1,
                       jobs = -1,
                       rng = None):
  rng = rng or np.random.default_rng (randint ())
  # First call to `_fit|fit_activations', creates the model if
  # `create_model' holds:
  abstr = from_discrete_abstraction \
    (abstr, create_model = not learn_model_structure)

  p1 (f'Projecting the training dataset...')
  x_ref = lazy_activations_on_indexed_data \
    (abstr.transform_activations, dnn, data, idxs,
     layer_indexes = [ fl.layer_index for fl in abstr.flayers ],
     pass_kwds = False)

  train_idxs = slice (None)
  if learn_model_structure:
    p1 (f'Learning abstract model structure...')
    learn_idxs = \
      rng.choice (len (x_ref),
                  min (len (x_ref),
                       int (learn_model_structure_train_ratio * len (x_ref))),
                  replace = False)
    abstr._fit (x_ref[learn_idxs], n_jobs = jobs)
    train_idxs = np.ones (len (x_ref), dtype = bool)
    train_idxs[learn_idxs] = False
    if not np.any (train_idxs):
      train_idxs = None

  if train_idxs is not None:
    p1 (f'Fitting the abstract model...')
    abstr._fit (x_ref[train_idxs], n_jobs = jobs)

  p1 (f'Computing reference probabilities...')
  P = dict (none = abstr.probability (x_ref, n_jobs = jobs))

  for fi, (fl, f) in enumerate (abstr.ordered_features):
    p_prime = np.asarray ([])
    for _n in range (iters):
      tp1 (f'Perturbing feature {f} of layer {fl} (iteration {_n+1}/{iters})...')
      x_prime = random_shift_feature (x_ref, fi, fl, f, rng = rng)
      p_prime = np.append (p_prime, abstr.probability (x_prime, n_jobs = jobs))
      del x_prime
    p1 (f'Feature {f} of layer {fl} done.')
    P[(fl, f)] = p_prime

  if return_dict:               # what else?
    P['none'] = np.tile (P['none'], iters)
    return P


def perturb_features (test_object,
                      # abstr = None,
                      abstraction = None,
                      iters = 1,
                      jobs = -1,
                      **_):
  abstr = DiscreteFAbstraction.from_file (abstraction_path (abstraction),
                                          dnn = test_object.dnn)
  return perturb_features_ (test_object.dnn, test_object.train_data,
                            abstr, iters = iters, jobs = jobs)

ap.set_defaults (cmd = perturb_features)

# ---

def attack_ (dnn, train_data, ref_data, abstr, attacks,
             from_discrete_abstraction = BNAbstraction.from_discrete_abstraction,
             learn_model_structure = False,
             learn_model_structure_train_ratio = .1,
             return_dict = False,
             idxs = slice (None),
             jobs = -1,
             rng = None):
  rng = rng or np.random.default_rng (randint ())
  # First call to `_fit|fit_activations', creates the model if
  # `create_model' holds:
  abstr = from_discrete_abstraction \
    (abstr, create_model = not learn_model_structure)

  train_idxs = slice (None)
  if learn_model_structure:
    p1 (f'Learning abstract model structure...')
    # from sklearn.model_selection import train_test_split
    # learn_idxs, train_idxs = \
    #   train_test_split (train_idxs,
    #                     train_size = min (len (train_data),
    #                                       int (learn_model_structure_train_ratio * len (train_data))),
    #                     random_state = randint (rng))
    learn_idxs = \
      rng.choice (len (train_data),
                  min (len (train_data),
                       int (learn_model_structure_train_ratio * len (train_data))),
                  replace = False)
    # print (type (train_data.data))
    # print (np.asarray(learn_idxs).shape)
    # print (np.asarray(learn_idxs).squeeze ().shape)
    # learn_idxs = np.zeros (len (train_data), dtype = bool)
    # learn_mask \
    #   [rng.choice (len (train_data),
    #                min (len (train_data),
    #                     int (learn_model_structure_train_ratio * len (train_data))),
    #                replace = False)] = True
    lazy_activations_on_indexed_data \
      (abstr.fit_activations, dnn, train_data, learn_idxs,
       layer_indexes = [ fl.layer_index for fl in abstr.flayers ],
       pass_kwds = False)
    train_idxs = np.ones (len (train_data), dtype = bool)
    train_idxs[learn_idxs] = False
    if not np.any (train_idxs):
      train_idxs = None

  if train_idxs is not None:
    p1 (f'Fitting the abstract model...')
    lazy_activations_on_indexed_data \
      (abstr.fit_activations, dnn, train_data, train_idxs,
       layer_indexes = [ fl.layer_index for fl in abstr.flayers ],
       pass_kwds = False)

  p1 (f'Projecting the reference dataset...')
  x_ref = lazy_activations_on_indexed_data \
    (abstr.transform_activations, dnn, ref_data, idxs,
     layer_indexes = [ fl.layer_index for fl in abstr.flayers ],
     pass_kwds = False)

  p1 (f'Computing reference probabilities...')
  P = dict (none = abstr.probability (x_ref, n_jobs = jobs))

  attacks_ = (attacks,) if isinstance (attacks, dict) else attacks
  Pl = []
  for attack_dict in attacks_:
    P_ = dict (**P)
    for attack in attack_dict:
      p1 (f'Projecting {attack} adversarial dataset...')
      x_adv = lazy_activations_on_indexed_data \
        (abstr.transform_activations, dnn, attack_dict[attack], idxs,
         layer_indexes = [ fl.layer_index for fl in abstr.flayers ],
         pass_kwds = False)
      P_[attack] = abstr.probability (x_adv, n_jobs = jobs)
    Pl.append (P_)

  if return_dict:               # what else?
    return Pl[0] if isinstance (attacks, dict) else Pl

# ---

ap_check = subparsers.add_parser ('check')
add_abstraction_arg (ap_check)
ap_check.add_argument ('--train-size', '-ts', type = int,
                       help = 'train dataset size (default is all)')
ap_check.add_argument ('--size', '-s', dest = 'test_size',
                       type = int, default = 100,
                       help = 'test dataset size (default is 100)')
ap_check.add_argument ('--summarize-probas', '-p', action = 'store_true',
                       help = 'fit the BN with all training data and then '
                       'assess the probability of the test dataset')

def check (test_object,
           abstraction = None,
           test_size = 100,
           train_size = None,
           summarize_probas = False,
           summarize_coverages = True,
           transformed_data = {},
           **_):
  rng = np.random.default_rng (randint ())
  tests = dict (raw = test_object.raw_data, **transformed_data)
  bn_abstr = BNAbstraction.from_file (test_object.dnn, abstraction_path (abstraction))

  if summarize_probas:
    fit_data_sample (test_object.dnn, bn_abstr, test_object.train_data, train_size, rng)
    probs = {
      t: eval_probas (test_object.dnn, bn_abstr, tests[t], test_size, rng)
      for t in tests
    }
    print (probs)

  if summarize_coverages:
    covs = {
      t: eval_coverages (test_object.dnn, bn_abstr, tests[t], test_size, rng)
      for t in tests
    }
    print (covs)

ap_check.set_defaults (cmd = check)

# ---

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
