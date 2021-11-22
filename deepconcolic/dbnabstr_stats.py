from .utils_io import sys, os, np, p1, h1, h2, tp1
from .utils_args import argparse, dispatch_cmd
from .utils_mp import init as mp_init, FFPool, forking, np_share
from .utils_funcs import randint, rng_seed
from .plotting import plt, pgf, pgf_setup, show
from fasteners import InterProcessLock
from itertools import product
from scipy.stats import wasserstein_distance, entropy
from scipy.linalg import norm
from scipy.spatial.distance import cityblock, chebyshev
from scipy.spatial.distance import euclidean, seuclidean
from scipy.spatial.distance import correlation, cosine
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

pgf_setup (**{
  'font.size': 9,
  'axes.linewidth': 1.,
  'axes.unicode_minus': True,           # fix mpl bug in 3.3.0?
  'axes.labelsize': 'small',
  'lines.linewidth': .2,
  'lines.markersize': 1.,
  'xtick.labelsize': 'x-small',
  'ytick.labelsize': 'x-small',
  'legend.fontsize': 'x-small',
  'legend.title_fontsize': 'small',
  'legend.handlelength': .5,
  'pgf.preamble': "\n".join([
    r"\usepackage[utf8x]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amssymb}",
    r"\usepackage{relsize}",
    r"\usepackage{xspace}",
    r"\usepackage[scientific-notation=true]{siunitx}",
    r'\renewcommand\P[1]{\ensuremath{\mathrm{Pr}\!\left(#1\right)}\xspace}',
  ])
})
pgfnumopt = '[round-mode=figures,round-precision=3]'

# ---

def read_pkl_ (i):
  try:
    return pd.read_pickle (i)
  except ValueError:
    import pickle5
    with open (i, "rb") as f:
      return pickle5.load (f)

def read_pkl (i, extra_ids = (), value_name = 'dist', discriminate_extended = True):
  dt = read_pkl_ (i)
  dt = dt.reset_index ()
  dt[r'\mathit{AF}'] = 1. - dt[r'R^2']
  dt = dt.melt (id_vars = _abstr_keys + ('model', 'dataset',) + extra_ids,
                value_name = value_name) \
         .rename (columns = {'techs': 'feature extraction'})
  dt['discretization'] = \
    dt.discr_strategies + \
    dt.num_intervals.astype (str) + \
    (np.where (dt.extended, '-X', '') if discriminate_extended else '')
  return dt

def read_npz (npzfile):
  data = np.load (npzfile)
  prob = lambda k: pd.DataFrame (data[k], columns = ('probability',)) \
                     .rename_axis ('input')
  ffeats = tuple (str (i) for i in range (len (data) - 1))
  feats = tuple (str ((i, j)) for i in range (1,4) for j in range (2))
  P = pd.concat ([ prob ('p_ref') ] +
                 [ prob (f'p_prime_feature{i}') for i in ffeats ],
                 axis = 'columns',
                 keys = ('none',) + feats,
                 names = ('perturbed feature',))
  return P, feats

def from_dict (P, name = 'perturbed feature'):
  prob = lambda k: pd.DataFrame (P[k], columns = ('probability',)) \
                     .rename_axis ('input')
  ffeats = tuple (str (i) for i in range (len (P) - 1))
  feats = tuple (k for k in P.keys () if k != 'none')
  sfeats = tuple (str (f) for f in feats)
  P = pd.concat ([ prob ('none') ] +
                 [ prob (k) for k in feats ],
                 axis = 'columns',
                 keys = ('none',) + sfeats,
                 names = (name,))
  return P, sfeats

def flat (a):
  return np.asarray (a.values).flatten ()

# ---

max_error = lambda Y1, Y2, **_: np.amax (np.abs (Y1 - Y2), axis = 0)
rmse = lambda *_, **__: mse (*_, **__, squared = False)

distances = (
  ('L_0', lambda P, Q: norm (P - Q, 0)),
  ('L_1', cityblock),
  ('L_2', euclidean),
  ('L_{∞}', chebyshev),
  # ('\mathit{Slater}', seuclidean),
  ('\mathit{corr}', correlation),
  ('\mathit{cos}', cosine),
  ('\mathit{KL}', lambda Pref, Pprime: entropy (Pprime, Pref)),
  ('\mathit{JS}', jensenshannon),
  ('W_1', wasserstein_distance),
  ('\mathit{MAX}', max_error),
  ('\mathit{MSE}', mse),
  ('\mathit{MSLE}', msle),
  ('\mathit{RMSE}', rmse),
  ('\mathit{MAPE}', mape),
  ('\mathit{MAE}', mae),
  ('R^2', r2_score),
  ('\mathit{AF}', lambda P, Q: 1. - r2_score (P, Q)),
)
included_dists_ = ((
  # r'L_0',
  r'L_1',
  r'L_2',
  r'L_{∞}',
),(
  # r'W_1',
  r'\mathit{JS}',
  # '\mathit{KL}',
  r'\mathit{corr}',
  r'\mathit{cos}',
),(
  # r'\mathit{MAX}',
  r'\mathit{MSE}',
  # r'\mathit{MSLE}',
  r'\mathit{RMSE}',
# ),(
#   r'\mathit{MAPE}',
  r'\mathit{MAE}',
  # r'R^2',
  r'\mathit{AF}',
))
flat_included_dists_ = [y for z in included_dists_ for y in z]
summary_dists_ = (
  r'L_2',
  r'\mathit{cos}',
  r'\mathit{AF}',
)

def compute_dists (P, feats, name = 'perturbed feature'):
  Pref = flat (P['none'])
  Pprm = { f: flat (P[f]) for f in feats }
  dists = { f: { k: d (Pref, Pprm[f]) for k, d in distances } for f in feats }
  return pd.DataFrame.from_dict (dists).rename_axis ('distance') \
                     .transpose ().rename_axis (name)

def compute_shuf_dists (P, n = 1):
  Pref = flat (P['none'])
  Pshufs = []
  for i in range (n):
    Pshufs.append (Pref.copy ())
    np.random.shuffle (Pshufs[-1])
  dists = [{ k: d (Pref, Pshuf) for k, d in distances }
           for Pshuf in Pshufs]
  return pd.DataFrame.from_records (dists)

def compute_half_half_dists (P, feats, name = 'perturbed feature'):
  Pprm = { f: flat (P[f]) for f in feats }
  dists = { f: { k: d (Pprm[f][:len (Pprm[f]) // 2],
                       Pprm[f][len (Pprm[f]) // 2:])
                 for k, d in distances } for f in feats }
  return pd.DataFrame.from_dict (dists).rename_axis ('distance') \
                     .transpose ().rename_axis (name)

def def_pp_dists (P, dists):
  def aux (g):
    g.set_titles (template = '')
    for col_val, ax in g.axes_dict.items ():
      text = lambda *_, fontsize = 8, **__: \
        ax.text (0.99, *_, fontsize = fontsize, ha = 'right', va = 'top',
                 transform = ax.transAxes, **__)
      fmt = lambda d: \
        (f'\\num{pgfnumopt}{{{d}}}' if d != np.inf else r'{∞}') if pgf \
        else f'{d:.3f}'
      dv, vv = .07, 0
      text (1., f'Perturbed feature: {col_val}', fontsize = 10)
      if col_val in dists.index:
        for i, d in enumerate (flat_included_dists_):
          text (1. - dv * (i + 1), f'$d_{{{d}}} = {fmt (dists[d][col_val])}$')
  return aux

# ---

def widen (P, xname = 'probability'):
  return P.melt (col_level = 0, value_name = xname)

def plot_probs (Pwide, feats, xname = 'probability', postproc = None, **_):
  g = sns.displot (data = Pwide, x = xname, col = 'perturbed feature', **_)
  if postproc is not None: postproc (g)
  return plt.gcf ()

def plot_dis (Pwide, bins, xname = 'probability', postproc = None, **_):
  g = sns.displot (data = Pwide, x = xname, bins = bins,
                   col = 'perturbed feature', **_)
  if postproc is not None: postproc (g)
  return plt.gcf ()

# ---

def diff (P, feats, xname = r'$P_{\mathit{ref}} - P_f$'):
  return pd.concat (tuple (P[f] - P['none'] for f in feats),
                    axis = 'columns', keys = feats,
                    names = ('perturbed feature',))

def plot_diffs (Pdiff, xname = r'$P_{\mathit{ref}} - P_f$', postproc = None, **_):
  # Pdiff = pd.concat (tuple (P[f] - P['none'] for f in feats),
  #                    axis = 'columns', keys = feats,
  #                    names = ('perturbed feature',))
  # xname = r'$P_{\mathit{ref}} - P_f$'
  # Pdiffwide = Pdiff.melt (col_level = 0, value_name = xname)
  g = sns.displot (data = widen (Pdiff, xname = xname),
                   x = xname, col = 'perturbed feature', **_)
  if postproc is not None: postproc (g)
  return plt.gcf ()

# ---

def tabulate_dists (texfile, dists, index = True):
  print (texfile)
  dists = dists[flat_included_dists_]
  dists.columns = '{\\(d_{' + dists.columns + '}\\)}'
  idxc = 'r' if index else ''
  dists.to_latex (buf = texfile,
                  column_format = f'{idxc} *{{{len (distances)}}}{{S{pgfnumopt}@{{~}}}}',
                  float_format = '{:.12f}'.format,
                  escape = False,
                  index = index)

# ---

def add_io_args (ap, input = 'npz'):
  if input == 'npz':
    ap.add_argument ('npz_input', metavar = 'NPZ',
                     help = 'input NPZ file with probability vectors')
  elif input == 'pkl':
    ap.add_argument ('pkl_input', metavar = 'PKL',
                     help = 'input NPZ file with distance statistics')
  elif input == 'pkls':
    ap.add_argument ('pkl_inputs', metavar = 'PKL', nargs = '+',
                     help = 'input NPZ files with distance statistics')
  ap.add_argument ('--outdir', '--output', '-O', metavar = 'DIR',
                   help = 'input directory for plots', default = os.path.curdir)
  ap.add_argument ('--prefixnames', metavar = 'STR', default = 'P',
                   dest = 'prefix',
                   help = 'prefix for output files')

parser = argparse.ArgumentParser \
  (description = 'Script for gathering and plotting stats about probabilistic'
   ' layer-abstraction models',
   formatter_class = argparse.ArgumentDefaultsHelpFormatter,
   epilog = '''By default, this program generates interactive plots.
   Set the DC_MPL_BACKEND environment variable to one of following to
   "png" or "pgf" to make it output persistent files in the directory
   given as argument to `--outdir' instead.  (Use "pgf" to get both
   PDG and PGF files).''')
subparsers = parser.add_subparsers (title = 'sub-commands', required = True,
                                    dest = 'cmd')

ap = subparsers.add_parser ('plot-dists')
add_io_args (ap)

def figdims (aspect = 1., totmargin = 30., ncol = 3):
  pgfwidth = (505.89 - totmargin) / 72.27
  pgfheight = pgfwidth / aspect
  return dict (**(dict (height = pgfheight / ncol) if pgf else {}),
               aspect = aspect)

def plot_dists (npz_input = None, outdir = None, prefix = None):
  # args = ap.parse_args ()
  # outdir = args.outdir
  figargs = figdims (aspect = 1.2)

  P, feats = read_npz (npz_input)
  Pwide = widen (P)

  n_bins = 50
  P_binned, bins = pd.cut (x = flat (P), bins = n_bins, retbins = True)

  print ('Computing distances...')

  dists = compute_dists (P, feats)
  pp_dists = def_pp_dists (P, dists)

  print (dists)

  # show (plot_probs (Pwide, feats, kind = 'kde', common_norm = False,
  #                   fill = True, alpha = .9, color = 'red',
  #                   col_order = feats + ('none',),
  #                   col_wrap = 3, **figargs),
  #       outdir = outdir, basefilename = prefix)

  # show (plot_diffs (diff (P, feats),
  #                   kind = 'kde', common_norm = False,
  #                   fill = True, alpha = .9,
  #                   col_wrap = 3, **figargs),
  #       outdir = outdir, basefilename = f'{prefix}diffs')

  show (plot_dis (Pwide, bins,
                  stat = 'density',
                  col_order = feats # + ('none',)
                  ,
                  col_wrap = 3, **figargs,
                  postproc = pp_dists),
        outdir = outdir, basefilename = f'{prefix}dens')

  # # hist, bin_edges = np.histogram (P['none'], bins = bins, density = True)
  # # Pref_binned, _ = np.histogram (P['none'], bins = bins)
  # # hist = lambda f: pd.DataFrame (np.histogram (P[f], bins = bins)[0],
  # #                                index = bins) \
  # #                    .rename_axis ('bin')
  # # print (len (P['none']))
  # # print (len (np.asarray (P['none'].values).flatten ()))
  # # print (len (pd.cut (np.asarray (P['none'].values).flatten (),
  # #                     bins = bins, labels = False)))
  # # sys.exit (0)
  # # hist = lambda f: pd.DataFrame (pd.cut (np.asarray (P[f].values).flatten (),
  # #                                        bins = bins, labels = False),
  # #                                columns = ('count',)) \
  # #                    .rename_axis ('bin')
  # # Phists = pd.concat (tuple (hist (f) for f in feats + ('none',)),
  # #                     axis = 'columns', keys = feats + ('none',),
  # #                     names = ('distributions',))
  # # print (Phists)
  # # print (np.histogram (P['none'], bins = bins)[0])
  # # print (sum (np.histogram (P['none'], bins = bins)[0]))
  # pd_bins = pd.IntervalIndex.from_breaks (bins)
  # Phists = pd.concat ([ pd.DataFrame (np.histogram (P[f], bins = bins)[0],
  #                                     columns = ('count',), index = pd_bins) \
  #                       .rename_axis ('bin') for f in ('none',) + feats ],
  #                     axis = 'columns',
  #                     keys = ('none',) + feats,
  #                     names = ('perturbed feature',))
  # print (Phists)
  # # print (hist.sum ())
  # # print (np.sum(hist * np.diff(bin_edges)))
  # # show \
  # #   (plot_counted (widen (Phists, xname = 'count'), kind = 'hist'),
  # #    outdir = outdir, basefilename = f'{prefix}hist')
  # from scipy.stats import wasserstein_distance
  # print ([wasserstein_distance (flat (P['none']),
  #                               flat (P[f]))
  #         for f in feats ])

  # Pshuff = flat (P['none']).copy ()
  # np.random.shuffle (Pshuff)
  # assert wasserstein_distance (flat (P['none']), Pshuff) == 0.

  tabulate_dists (f'{prefix}distances.tex',
                  dists)
  # tabulate_dists (f'{prefix}shuf-distances.tex',
  #                 compute_shuf_dists (P, 10),
  #                 index = False)
  # tabulate_dists (f'{prefix}half-half-distances.tex',
  #                 compute_half_half_dists (P, feats))

ap.set_defaults (cmd = plot_dists)

# ---

_defaults = {
  'techs': ('pca', 'ica'),
  'nfeats': (3,),
  'discr_strategies': ('uniform', 'quantile'),
  'num_intervals': (5, 10,),
  'extended': (False, True),
  'train_size': 20000,          # for layer-wise abstraction only
  'iters': 10,
  'nlayers': (3,),
}

all_tests = [                   # hardcoded test for now;
  {
    'dataset': 'mnist',
    'model': '~/work/repos/DeepConcolic/saved_models/mnist_complicated.h5',
    'layers': ('max_pooling2d_1', 'max_pooling2d_2', 'activation_5'),
  },
  {
    'dataset': 'cifar10',
    'model': '~/work/repos/DeepConcolic/saved_models/cifar10_complicated.h5',
    'layers': ('max_pooling2d_1', 'max_pooling2d_2', 'activation_5'),
  },
]

_abstr_keys = (
  'nlayers',
  'techs',
  'nfeats',
  'discr_strategies',
  'num_intervals',
  'extended',
)

ap = subparsers.add_parser ('basic-dists')
ap.add_argument ('--iters', '-N', type = int, default = 10)
ap.add_argument ('--jobs', '-j', type = int, default = 5)
ap.add_argument ('--dry-run', '-n', action = 'store_true')

def basic_dists (iters = 10, jobs = 1, dry_run = False, repeats = 1, num_workers = 5):
  allf = 'all_dists.pkl'
  hhf = 'all_hhdists.pkl'

  for t in all_tests:
    get = lambda k: t.get (k, _defaults.get (k))

    dataset, model = get ('dataset'), get ('model')
    layers, train_size = get ('layers'), get ('train_size')

    h1 (f'Considering {model}')
    olock = InterProcessLock ('.lock')

    def dotest (test_object, abstr_info, pid = 0):
      from .dbnabstr import create, perturb_features_
      nlayers, tech, nfeats, discr_strategy, num_intervals, extended = abstr_info

      all_flat = None
      with olock:
        if os.path.exists (allf): all_flat = read_pkl_ (allf).reset_index ()

      if all_flat is not None and \
         len (all_flat[(all_flat['dataset'] == dataset) &
                       (all_flat['model'] == model) &
                       (all_flat['nlayers'] == nlayers) &
                       (all_flat['techs'] == tech) &
                       (all_flat['nfeats'] == nfeats) &
                       (all_flat['discr_strategies'] == discr_strategy) &
                       (all_flat['num_intervals'] == num_intervals) &
                       (all_flat['extended'] == extended)]) >= repeats:
        p1 (f'Skipping {str (abstr_info)}')
        return True

      h2 (str ((dataset,) + abstr_info))
      if dry_run:
        return False

      os.mkdir (f'./{pid}')
      os.chdir (f'./{pid}')

      dr = f'{tech}({nfeats})::{tech}({nfeats})'
      abstr = create (test_object, train_size = get ('train_size'),
                      abstraction_only = True,
                      dump_abstraction = False,
                      return_abstraction = True,
                      layers = layers[:nlayers],
                      dimred_specs = dr,
                      num_intervals = num_intervals,
                      discr_strategy = discr_strategy,
                      extended_discr = extended)

      for _ in range (repeats):
        P = perturb_features_ (test_object.dnn, test_object.train_data,
                               abstr, return_dict = True,
                               iters = iters, jobs = jobs)

        def assign_keys (dists):
          for k, i in zip (_abstr_keys + ('model', 'dataset'),
                           abstr_info + (model, dataset)):
            dists = dists.assign (**{ k: i # for k, i in zip (_abstr_keys, abstr_info)
                                     }) \
                         .set_index (k, append = True)
          return dists

        Pf = from_dict (P)

        dists = compute_dists (*Pf)
        dists = assign_keys (dists)
        hhdists = compute_half_half_dists (*Pf)
        hhdists = assign_keys (hhdists)

        os.chdir ('..')
        os.rmdir (f'./{pid}')

        # XXX: very hack-ish!
        with olock:
          all_dists = read_pkl_ (allf) if os.path.exists (allf) else None
          all_dists = dists if all_dists is None else all_dists.append (dists)
          all_dists.to_pickle (allf)

          # Only valid if "symmetric" vector of reference inputs (e.g
          # X_test repeated once).
          all_hhdists = read_pkl_ (hhf) if os.path.exists (hhf) else None
          all_hhdists = hhdists if all_hhdists is None else all_hhdists.append (hhdists)
          all_hhdists.to_pickle (hhf)
      return True

    def make_worker (model, **__):
      rng_seed (np.random.default_rng ().integers (2**32-1))
      from .utils import test_objectt
      from .dbnabstr import load_dataset, load_model
      test_object = test_objectt (load_model (os.path.expanduser (model)),
                                  *load_dataset (dataset))
      def aux (*_):
        return dotest (test_object, *_, **__)
      return aux
    ffpool_args = make_worker, model

    pool = FFPool (*ffpool_args, processes = num_workers, verbose = True, pass_pid = True)
    pool.start ()

    cnt = 0
    for abstr_info in product (*(get (k) for k in _abstr_keys)):
      cnt += 1
      pool.put (abstr_info)

    while cnt > 0:
      pool.get ()
      cnt -= 1

    # terminate pool:
    pool.join ()

    del olock

ap.set_defaults (cmd = basic_dists)

# ---

ap = subparsers.add_parser ('gen-attacks')
ap.add_argument ('--dtsuff', type = str, default = 'a')

def gen_attacks (dtsuff = ''):
  from .utils import test_objectt
  from .dbnabstr import load_dataset, load_model
  from .eval_classifier import _clr
  from art.attacks.evasion import FastGradientMethod
  from art.attacks.evasion import ProjectedGradientDescent
  from art.attacks.evasion import CarliniL0Method
  from art.attacks.evasion import CarliniLInfMethod
  from art.attacks.evasion import CarliniL2Method
  from art.attacks.evasion import DeepFool

  rng_seed (np.random.default_rng ().integers (2**32-1))

  for t in all_tests:
    get = lambda k: t.get (k, _defaults.get (k))

    dataset, model = get ('dataset'), get ('model')
    test_object = test_objectt (load_model (os.path.expanduser (model)),
                                *load_dataset (dataset))
    model = _clr (test_object.dnn)

    preds = np.argmax (model.predict (test_object.raw_data.data), axis = 1)
    acc = np.sum (preds == test_object.raw_data.labels) / len (test_object.raw_data)
    p1 ("Test accuracy: %.2f%%" % (acc * 100))

    pgdargs = dict (batch_size = 128)
    cwargs = dict (confidence = .1, batch_size = 128, verbose = True)
    attacks = dict (fgsm = FastGradientMethod (model, eps = .1),
                    pgdlinf = ProjectedGradientDescent (model, norm = 'inf',
                                                        eps = .1, eps_step = .01,
                                                        **pgdargs),
                    pgdl2 = ProjectedGradientDescent (model, norm = 2,
                                                      eps = 10, eps_step = 1,
                                                      **pgdargs),
                    # cwl0 = CarliniL0Method (model, **cwargs),
                    cwlinf = CarliniLInfMethod (model, **cwargs, eps = .1),
                    cwl2 = CarliniL2Method (model, **cwargs),
                    deepfool = DeepFool (model, batch_size = 128))
    X_adv = {}
    for a in attacks:
      try:
        h2 (f'{dataset}, {a}')
        attack = attacks[a]
        x_adv = attack.generate (x = test_object.raw_data.data)

        # Evaluate the classifier on the adversarial examples
        preds = np.argmax (model.predict (x_adv), axis = 1)
        acc = np.sum (preds == test_object.raw_data.labels) / len (test_object.raw_data)
        p1 ("Test accuracy on adversarial samples: %.2f%% with %s attack" % (acc * 100, a))

        X_adv[a] = x_adv
      except KeyboardInterrupt:
        p1 (f'Skipping {a}')

    np.savez_compressed (f'{dataset}-{dtsuff}.npz', attacks = X_adv, num_gens = 1)

ap.set_defaults (cmd = gen_attacks)

# ---

ap = subparsers.add_parser ('attack-dists')
ap.add_argument ('--jobs', '-j', type = int, default = 5)
ap.add_argument ('--dry-run', '-n', action = 'store_true')

def attack_dists (jobs = 1, dry_run = False, repeats = 1, num_workers = 4):

  dtsuffs = ['a', 'b', 'c', 'd', 'e']

  for t in all_tests:
    get = lambda k: t.get (k, _defaults.get (k))

    dataset, model = get ('dataset'), get ('model')
    layers, train_size = get ('layers'), get ('train_size')

    # if not os.path.exists (f'{dataset}{dtsuff}.npz') and not dry_run:
    #   gen_attacks (dtsuff)

    # if not os.path.exists (f'{dataset}{dtsuff}.npz'):
    #   continue

    h1 (f'Considering {model}')
    olock = InterProcessLock ('.lock')

    def dotest (test_object, attacks, abstr_info, pid = 0):
      from .dbnabstr import create, attack_
      nlayers, tech, nfeats, discr_strategy, num_intervals, extended = abstr_info

      allf = f'{dataset}_dists.pkl'
      hhf = f'{dataset}_hhdists.pkl'

      all_flat = None
      with olock:
        if os.path.exists (allf): all_flat = read_pkl_ (allf).reset_index ()

      if all_flat is not None and \
         all (len (all_flat[(all_flat['dataset'] == dataset) &
                            (all_flat['dtsuff'] == dtsuff) &
                            (all_flat['model'] == model) &
                            (all_flat['nlayers'] == nlayers) &
                            (all_flat['techs'] == tech) &
                            (all_flat['nfeats'] == nfeats) &
                            (all_flat['discr_strategies'] == discr_strategy) &
                            (all_flat['num_intervals'] == num_intervals) &
                            (all_flat['extended'] == extended)]) >= repeats
              for dtsuff in dtsuffs):
        p1 (f'Skipping {str (abstr_info)}')
        return True

      h2 (str ((dataset,) + abstr_info))
      if dry_run:
        return False

      os.makedirs (f'./{pid}', exist_ok = True)
      os.chdir (f'./{pid}')

      dr = f'{tech}({nfeats})::{tech}({nfeats})'
      abstr = create (test_object, get ('train_size'),
                      abstraction_only = True,
                      dump_abstraction = False,
                      return_abstraction = True,
                      layers = layers[:nlayers],
                      dimred_specs = dr,
                      num_intervals = num_intervals,
                      discr_strategy = discr_strategy,
                      extended_discr = extended)

      for _ in range (repeats):
        Ps = attack_ (test_object.dnn, test_object.train_data, test_object.raw_data,
                      abstr, attacks, return_dict = True, jobs = jobs)

        def assign_keys (dists, dtsuff):
          for k, i in zip (_abstr_keys + ('model', 'dataset', 'dtsuff'),
                           abstr_info + (model, dataset, dtsuff)):
            dists = dists.assign (**{ k: i }).set_index (k, append = True)
          return dists

        dists, hhdists = None, None
        for P, dtsuff in zip (Ps, dtsuffs):
          Pf = from_dict (P, name = 'attack')

          dists_ = compute_dists (*Pf, name = 'attack')
          dists_ = assign_keys (dists_, dtsuff)
          dists = dists_ if dists is None else dists.append (dists_)

          hhdists_ = compute_half_half_dists (*Pf, name = 'attack')
          hhdists_ = assign_keys (hhdists_, dtsuff)
          hhdists = hhdists_ if hhdists is None else hhdists.append (hhdists_)

        os.chdir ('..')
        os.rmdir (f'./{pid}')

        # XXX: very hack-ish!
        with olock:
          all_dists = read_pkl_ (allf) if os.path.exists (allf) else None
          all_dists = dists if all_dists is None else all_dists.append (dists)
          all_dists.to_pickle (allf)

          # Only valid if "symmetric" vector of reference inputs (e.g
          # X_test repeated once).
          all_hhdists = read_pkl_ (hhf) if os.path.exists (hhf) else None
          all_hhdists = hhdists if all_hhdists is None else all_hhdists.append (hhdists)
          all_hhdists.to_pickle (hhf)
      return True

    def make_worker (model, dataset, dtsuffs, **__):
      rng_seed (np.random.default_rng ().integers (2**32-1))
      from .utils import test_objectt
      from .dbnabstr import load_dataset, load_model
      test_object = test_objectt (load_model (os.path.expanduser (model)),
                                  *load_dataset (dataset))
      attacks = []
      for dtsuff in dtsuffs:
        p1 (f'Loading {dataset}-{dtsuff}.npz...')
        dct = np.load (f'{dataset}-{dtsuff}.npz', allow_pickle = True)
        attacks.append (dct['attacks'].reshape ((1,))[0])
      def aux (*_):
        return dotest (test_object, attacks, *_, **__)
      return aux

    pool = FFPool (make_worker, model, dataset, dtsuffs,
                   processes = num_workers, verbose = True, pass_pid = True)
    pool.start ()

    cnt = 0
    for abstr_info in product (*(get (k) for k in _abstr_keys)):
      cnt += 1
      pool.put (abstr_info)

    while cnt > 0:
      pool.get ()
      cnt -= 1

    # terminate pool:
    pool.join ()

    del olock

ap.set_defaults (cmd = attack_dists)

# ---

ap = subparsers.add_parser ('plot-basic-dists')
add_io_args (ap, 'pkl')


def plot_basic_dists (pkl_input = None, outdir = None, prefix = None):
  dists = read_pkl (pkl_input, extra_ids = ('perturbed feature',))
  figargs = figdims (ncol = 3)
  print (dists)
  # sns.catplot (data = dists, col = 'distance', y = 'dist', col_wrap = 3,
  #              kind = 'violin', cut = 0, bw = .9,
  #              sharey = False, hue = 'techs', x = 'num_intervals', **figargs)
  # sns.catplot (data = dists, col = 'distance', y = 'dist', col_wrap = 3,
  #              # kind = 'point', # markers=["^", "o"], linestyles=["-", "--"],
  #              kind = 'box', # markers=["^", "o"], linestyles=["-", "--"],
  #              sharey = False, hue = 'nfeats', x = 'discretization', **figargs)
  for dataset in dists.dataset.unique ():
    for i, included_dists in enumerate (included_dists_):
      dt = dists[(dists.dataset == dataset) &
                 (dists.nlayers == 3) &
                 (dists.nfeats == 3) &
                 (dists.num_intervals != 20) &
                 (dists.distance.isin (included_dists))].copy ()
      dt['distance'] = '$' + dt.distance + '$'
      g = sns.catplot (data = dt, hue = 'discretization',
                       x = 'perturbed feature', y = 'dist', # col_wrap = 3,
                       kind = 'point', # markers=["^", "o"], linestyles=["-", "--"],
                       # kind = 'box', # markers=["^", "o"], linestyles=["-", "--"],
                       sharey = 'row', sharex = True,
                       row = 'distance', col = 'feature extraction', **figargs,
                       s = .1,
                       legend = i == 0,
                       legend_out = False)
      # g = sns.relplot (data = dt, hue = 'discretization',
      #                  x = 'perturbed feature', y = 'dist', # col_wrap = 3,
      #                  kind = 'line', # markers=["^", "o"], linestyles=["-", "--"],
      #                  # kind = 'box', # markers=["^", "o"], linestyles=["-", "--"],
      #                  # sharey = True, sharex = False,
      #                  col = 'nfeats', row = 'techs', **figargs# ,
      #                  # legend_out = True
      #                  )
      # g.set_ylabels (d)
      # plt.legend (# loc = 'best',
      #             ncol = 3)
      # g.fig.legend(labels = [], ncol = 3, bbox_to_anchor = (1,1))
      g.set_titles (row_template = '{row_var}: {row_name}',
                    col_template = '{col_var}: {col_name}')
      g.set_xticklabels (rotation = 90)
      show (plt.gcf (),
            outdir = '.',
            basefilename = f'{prefix}-{dataset}-base-dists-{i}')
  #   # 'techs', 'nfeats',
  #   # ('num_intervals', 'discr_strategies', 'extended'),
  # # ]))

  # print (dists.melt (# id_vars = ['perturbed feature'], col_level = 0
  # ))

ap.set_defaults (cmd = plot_basic_dists)

# ---

ap = subparsers.add_parser ('plot-attack-dists')
add_io_args (ap, 'pkls')

def plot_attack_dists (pkl_inputs = None, outdir = None, prefix = None):
  figargs = figdims (ncol = 3)
  for pkl_input in pkl_inputs:
    dists = read_pkl (pkl_input, extra_ids = ('dtsuff', 'attack',))
    print (dists)
    # print (dists.index)
    # print (dists.columns)
    # dists = dists.pivot (index = [s for s in dists.columns if s not in ('distance', 'dist')],
    #                      # ('nlayers',
    #                      #  'feature extraction',
    #                      #  'nfeats',
    #                      #  'discr_strategies',
    #                      #  'num_intervals',
    #                      #  'extended',
    #                      #  'discretization',
    #                      #  'model', 'dataset','dtsuff', 'attack',),
    #                      columns = 'distance', values = 'dist')
    # print (dists.columns)
    for dataset in dists.dataset.unique ():
      for i, included_dists in enumerate (included_dists_):
        # included_dists = tuple (f'd_{{{d}}}' for d in included_dists)
        dt = dists[(dists.dataset == dataset) &
                   (dists.nlayers == 3) &
                   (dists.nfeats == 3) &
                   (dists.num_intervals != 20) &
                   (dists.distance.isin (included_dists))].copy ()
        # dt = dt.pivot (index = [s for s in dt.columns if s not in ('dist', 'distance')],
        #                columns = ['distance'],
        #                values = ['dist'])
        print (', '.join (dt.distance.unique ()))
        dt['distance'] = '$' + dt.distance + '$'
        g = sns.catplot (data = dt, hue = 'discretization',
                         x = 'attack', y = 'dist', kind = 'bar',
                         sharey = 'row', sharex = True,
                         row = 'distance',
                         col = 'feature extraction', **figargs,
                         # s = .1,
                         legend = True,
                         legend_out = False)
        g.set_titles (template = 'measure $p =$ {row_name} | feat. extr.: {col_name}')
        g.set_xticklabels (rotation = 90)
        g.set_ylabels (r'$d_p\left(\P{X_{\mathit{test}}\in\mathcal B}, \P{X_{\mathsf{attack}}\in\mathcal B}\right)$')
        # Some kind of table pivoting beforehand may be better, as
        # that's a bit hackish:
        # for ax, d in zip (g.axes[:,0], dt.distance.unique ()):
        #   ax.set_ylabel (d, rotation = 0)
        show (plt.gcf (),
              outdir = outdir,
              basefilename = f'{prefix}-{dataset}-attack-dists-{i}')

ap.set_defaults (cmd = plot_attack_dists)

# ---

ap = subparsers.add_parser ('plot-attack-summary')
add_io_args (ap, 'pkls')

def plot_attack_summary (pkl_inputs = None, outdir = None, prefix = None):
  dists = None
  for pkl_input in pkl_inputs:
    dists_ = read_pkl (pkl_input, extra_ids = ('dtsuff', 'attack',),
                       discriminate_extended = False)
    dists = dists_ if dists is None else dists.append (dists_)

  print (dists)
  # included_dists = tuple (f'd_{{{d}}}' for d in included_dists)
  for dataset in dists.dataset.unique ():
    dt = dists[(dists.dataset == dataset) &
               (dists.nlayers == 3) &
               (dists.nfeats == 3) &
               (dists.num_intervals != 20) &
               (dists.distance.isin (summary_dists_))].copy ()
    dt['distance'] = '$' + dt.distance + '$'
    figargs = figdims (aspect = 1.2, ncol = len (dt.attack.unique ()),
                       totmargin = 10.)
    g = sns.catplot (data = dt,
                     hue = 'discretization',
                     x = 'feature extraction',
                     y = 'dist',
                     kind = 'bar',
                     sharey = 'row',
                     sharex = True,
                     row = 'distance',
                     col = 'attack',
                     **figargs,
                     legend = False)
    # g.despine (left = True)
    if dataset == 'mnist':
      g.axes[0,3].legend (loc='upper left')
      g.set_xlabels (r'\phantom{feature extraction}')
    g.set_titles (template = '$p =$ {row_name} | {col_name}')
    # g.set_xticklabels (rotation = 90)
    g.set_ylabels (r'\strut')
    # g.set_ylabels (r'$d_p\left(\P{X_{\mathit{test}}\in\mathcal B}, \P{X_{\mathsf{attack}}\in\mathcal B}\right)$')
    # Some kind of table pivoting beforehand may be better, as
    # that's a bit hackish:
    # for ax, d in zip (g.axes[:,0], dt.distance.unique ()):
    #   ax.set_ylabel (d, rotation = 0)
    g.axes[1,0].set_ylabel (r'$d_p\left(\P{X_{\mathit{test}}\in\mathcal B}, \P{X_{\mathsf{attack}}\in\mathcal B}\right)$')
    show (plt.gcf (),
          outdir = outdir,
          basefilename = f'{prefix}-{dataset}-attack-summary')

ap.set_defaults (cmd = plot_attack_summary)

# ---

def main ():
  dispatch_cmd (parser = parser)

if __name__=="__main__":
  mp_init ()
  main ()

# ---
