import warnings
from typing import *
from utils import *
from engine import *
import numpy as np

from functools import reduce
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, Binarizer
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import log_loss, classification_report
from pomegranate import Node, BayesianNetwork
from pomegranate.distributions import (DiscreteDistribution,
                                       ConditionalProbabilityTable,
                                       JointProbabilityTable)

# ---


Interval = Tuple[Optional[float], Optional[float]]

def interval_dist (interval: Interval, v: float):
  interval = (interval[0] if interval[0] is not None else -np.inf,
              interval[1] if interval[1] is not None else np.inf)
  dist = np.amin (np.abs (interval - np.array(v)))
  dist = - dist if interval[0] < v < interval[1] else dist
  return dist

def interval_repr (interval: Interval, prec = 3, float_format = 'g'):
  interval = (interval[0] if interval[0] is not None else -np.inf,
              interval[1] if interval[1] is not None else np.inf)
  return '{lop}{:.{prec}{float_format}}, {:.{prec}{float_format}}{rop}' \
         .format (*interval,
                  prec = prec, float_format = float_format,
                  lop = '(' if interval[0] == -np.inf else '[',
                  rop = ')' if interval[1] == np.inf else ']')


# ---


class FeatureDiscretizer:

  @abstractmethod
  def feature_parts (self, feature: int) -> int:
    raise NotImplementedError

  @abstractmethod
  def edges (self, feature: int, value: float) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def part_edges (self, feature: int, part: int) -> Interval:
    raise NotImplementedError

  @abstractmethod
  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
    raise NotImplementedError

  @abstractmethod
  def get_params (self, deep = True) -> dict:
    raise NotImplementedError


class FeatureBinarizer (FeatureDiscretizer, Binarizer):

  def feature_parts (self, _feature):
    return 2

  def edges (self, feature: int, value: float) -> Interval:
    thr = self.threshold[0, feature]
    return (thr, np.inf) if value >= thr else (-np.inf, thr)

  def part_edges (self, feature: int, part: int) -> Interval:
    thr = self.threshold[0, feature]
    return (-np.inf, thr) if part == 0 else (thr, np.inf)

  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
    self.threshold = \
          feat_extr.transform (np.zeros (shape = x[:1].shape)) \
          .reshape (1, -1)
    self.fit (y)


class KBinsFeatureDiscretizer (FeatureDiscretizer, KBinsDiscretizer):

  def feature_parts (self, feature):
    return self.n_bins_[feature]

  def edges (self, feature: int, value: float) -> Interval:
    edges = np.concatenate((np.array([-np.inf]),
                            self.bin_edges_[feature][1:-1],
                            np.array([np.inf])))
    part = np.searchsorted (edges, value, side = 'right')
    return edges[part-1], edges[part]

  def part_edges (self, feature: int, part: int) -> Interval:
    edges = self.bin_edges_[feature]
    return ((-np.inf if part   == 0           else edges[part  ],
             np.inf  if part+2 == len (edges) else edges[part+1]))

  def fit_wrt (self, x, y, feat_extr, **kwds) -> None:
      super().fit (y)

  def get_params (self, deep = True) -> dict:
    p = super().get_params (deep)
    p['extended'] = False
    return p



class KBinsNOutFeatureDiscretizer (KBinsFeatureDiscretizer):

  def __init__(self, n_bins = 2, **kwds):
    super().__init__(n_bins = None if n_bins is None else max(2, n_bins), **kwds)
    self.one = n_bins == 1

  def feature_parts (self, feature):
    return self.n_bins_[feature] + 2 if not self.one else 3

  def edges (self, feature: int, value: float) -> Interval:
    edges = self.bin_edges_[feature]
    part = np.searchsorted (edges, value, side = 'right')
    return self.part_edges (feature, part)

  def part_edges (self, feature: int, part: int) -> Interval:
    edges = self.bin_edges_[feature]
    if self.one and part == 0:
      return (-np.inf, edges[0])
    elif self.one and part == 1:
      return (edges[0], edges[2])
    elif self.one and part == 2:
      return (edges[2], np.inf)
    else:
      return ((-np.inf if part == 0           else edges[part-1],
               np.inf  if part == len (edges) else edges[part  ]))

  def transform (self, x):
    return super ().transform (x) + 1

  def get_params (self, deep = True) -> dict:
    p = super().get_params (deep)
    p['extended'] = True
    return p

# ---


class BFcLayer (CoverableLayer):
  """
  Base class for layers to be covered by BN-based criteria.
  """

  def __init__(self, transform = None, discretization: FeatureDiscretizer = None, **kwds):
    super().__init__(**kwds)
    assert isinstance (discretization, FeatureDiscretizer)
    self.transform = transform
    self.discr = discretization

  def get_params (self, deep = True):
    return dict (name = self.layer.name,
                 transform = self.transform.get_params (deep))

  @property
  def num_features (self):
    '''
    Number of extracted features for the layer.
    '''
    return len (self.transform[-1].components_)


  def range_features (self):
    '''
    Range over all feature indexes.
    '''
    return range (self.num_features)


  def flatten_map (self, map, acc = None):
    x = np.vstack([e.flatten () for e in map[self.layer_index]])
    acc = np.hstack ((acc, x)) if acc is not None else x
    if acc is not x: del x
    return acc


  def dimred_activations (self, acts, acc = None):
    x = np.vstack([a.flatten () for a in acts[self.layer_index]])
    y = self.transform.transform (x)
    acc = np.hstack ((acc, y)) if acc is not None else y
    del x
    if acc is not y: del y
    return acc


  def dimred_n_discretize_activations (self, acts, acc = None):
    x = np.vstack([a.flatten () for a in acts[self.layer_index]])
    y = self.discr.transform (self.transform.transform (x))
    acc = np.hstack ((acc, y.astype (int))) if acc is not None else y.astype (int)
    del x, y
    return acc


# ---



def bayes_node_name(fl, idx):
  return '.'.join ((str(fl), *((str(i) for i in idx))))



class DiscretizedFeatureNode (Node):

  def __init__(self, flayer: BFcLayer, feature: int, *args, **kwds):
    super().__init__ (*args, name = bayes_node_name (flayer, (feature,)), **kwds)
    self.flayer = flayer
    self.feature = feature


  def discretized_range(self) -> range:
    return range (self.flayer.discr.feature_parts (self.feature))


  def interval(self, feature_interval: int) -> Interval:
    return self.flayer.discr.part_edges (self.feature, feature_interval)



class DiscretizedInputFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, **kwds):
    n = flayer.discr.feature_parts (feature)
    super().__init__(flayer, feature,
                     DiscreteDistribution ({ fbin: 0.0 for fbin in range (n) }),
                     **kwds)


class DiscretizedHiddenFeatureNode (DiscretizedFeatureNode):

  def __init__(self, flayer, feature, prev_nodes, **kwds):
    prev_nodes = list (pn for pn in prev_nodes
                       if len (pn.discretized_range ()) > 1)
    prev_distrs = [ n.distribution for n in prev_nodes ]
    prev_fparts = list ([ bin for bin in pn.discretized_range () ]
                        for pn in prev_nodes)
    n = flayer.discr.feature_parts (feature)
    if prev_fparts == [] or n == 1:
      del prev_distrs, prev_fparts
      self.prev_nodes_ = []
      super().__init__(flayer, feature,
                       DiscreteDistribution ({ fbin: 0.0 for fbin in range (n) }),
                       **kwds)
    else:
      condprobtbl = [ list (p) + [0.0] for p in product (*prev_fparts, range (n)) ]
      del prev_fparts
      self.prev_nodes_ = prev_nodes
      super().__init__(flayer, feature,
                       ConditionalProbabilityTable (condprobtbl, prev_distrs),
                       **kwds)
      del condprobtbl


# ---


class _BaseBFcCriterion (Criterion):
  '''
  ...

  - `feat_extr_train_size`: gives the proportion of training data from
    `bn_abstr_train_size` to use for feature extraction if <= 1;
    `min(feat_extr_train_size, bn_abstr_train_size)` will be used
    otherwise.

  ...
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               *args,
               epsilon = None,
               bn_abstr_train_size = None,
               bn_abstr_test_size = None,
               bn_abstr_n_jobs = None,
               feat_extr_train_size = 1,
               print_classification_reports = True,
               score_layer_likelihoods = False,
               report_on_feature_extractions = None,
               close_reports_on_feature_extractions = None,
               assess_discretized_feature_probas = False,
               outdir: OutputDir = None,
               **kwds):
    assert (print_classification_reports is None or isinstance (print_classification_reports, bool))
    assert (report_on_feature_extractions is None or callable (report_on_feature_extractions))
    assert (close_reports_on_feature_extractions is None or callable (close_reports_on_feature_extractions))
    assert (feat_extr_train_size > 0)
    self.epsilon = epsilon or 1e-4
    self.bn_abstr_n_jobs = bn_abstr_n_jobs
    self.bn_abstr_params = dict (train_size = bn_abstr_train_size or 0.5,
                                 test_size = bn_abstr_test_size or 0.5)
    self.feat_extr_train_size = feat_extr_train_size
    self.print_classification_reports = print_classification_reports
    self.score_layer_likelihoods = score_layer_likelihoods
    self.report_on_feature_extractions = report_on_feature_extractions
    self.close_reports_on_feature_extractions = close_reports_on_feature_extractions
    self.assess_discretized_feature_probas = assess_discretized_feature_probas
    self.flayers = list (filter (lambda l: isinstance (l, BFcLayer), clayers))
    clayers = list (filter (lambda l: isinstance (l, BoolMappedCoverableLayer), clayers))
    assert (clayers == [])
    self.base_dimreds = None
    self.outdir = outdir or OutputDir ()
    super().__init__(*args, **kwds)
    self._reset_progress ()


  def finalize_setup(self):
    self.analyzer.finalize_setup (self.flayers)


  def flatten_for_layer (self, map, fls = None):
    acc = None
    for fl in self.flayers if fls is None else fls:
      acc = fl.flatten_map (map, acc = acc)
    return acc


  def dimred_activations (self, acts, fls = None):
    acc = None
    for fl in self.flayers if fls is None else fls:
      acc = fl.dimred_activations (acts, acc = acc)
    return acc


  def dimred_n_discretize_activations (self, acts, fls = None):
    acc = None
    for fl in self.flayers if fls is None else fls:
      acc = fl.dimred_n_discretize_activations (acts, acc = acc)
    return acc


  # ---


  def reset (self):
    super().reset ()
    self.base_dimreds = None
    assert (self.num_test_cases == 0)
    self.outdir.reset_stamp ()
    self._reset_progress ()


  def fit_activations (self, acts):
    # Assumes `num_test_cases' has already been updated with the
    # inputs that triggered the given activations; otherwise, set
    # inertia to 0.0, which basically erases history.
    facts = self.dimred_n_discretize_activations (acts)
    nbase = self.num_test_cases - len (facts)
    self.N.fit (facts,
                inertia = (nbase / self.num_test_cases if nbase >= 0 else 0.0),
                n_jobs = int (self.bn_abstr_n_jobs))


  def register_new_activations (self, acts) -> None:
    # Take care `num_test_cases` has already been updated:
    self.fit_activations (acts)

    # Append feature values for new tests
    new_dimreds = self.dimred_activations (acts)
    self.base_dimreds = (np.vstack ((self.base_dimreds, new_dimreds))
                         if self.base_dimreds is not None else new_dimreds)
    if self.base_dimreds is not new_dimreds: del new_dimreds


  # ---


  def tests_feature_values_n_intervals (self, feature: int):
    fl, feature = self.fidx2fli[feature]
    dimreds = self.base_dimreds[..., feature : feature + 1].flatten ()
    intervals = [ (i, fl.discr.edges (feature, v)) for i, v in enumerate(dimreds) ]
    return dimreds, intervals


  def all_tests_close_to (self, feature: int, feature_interval: int):
    dimreds, intervals = self.tests_feature_values_n_intervals (feature)
    feature_node = self.N.states[feature]
    fl, flfeature = feature_node.flayer, feature_node.feature
    target_interval = fl.discr.part_edges (flfeature, feature_interval)
    all = [ (i, interval_dist (target_interval, dimreds[i])) for i, _ in intervals ]
    all.sort (key = lambda x: x[1])
    # np.random.shuffle (all)
    del dimreds, intervals
    return all


  # def _check_within (self, feature: int, expected_interval: int, verbose = True):
  #   def aux (t: Input) -> bool:
  #     acts = self.analyzer.eval (t, allow_input_layer = False)
  #     facts = self.dimred_n_discretize_activations (acts)
  #     res = facts[0][feature] == expected_interval
  #     if verbose and not res:
  #       dimred = self.dimred_activations (acts)
  #       dimreds = dimred[..., feature : feature + 1].flatten ()
  #       tp1 ('| Got interval {}, expected {} (fval {})'
  #            .format(facts[0][feature], expected_interval, dimreds))
  #     return res
  #   return aux


  # ----


  def _reset_progress (self):
    self.progress_file = self.outdir.stamped_filepath ( \
      str (self) + '_' + str (self.metric) + '_progress', suff = '.csv')
    write_in_file (self.progress_file,
                   '# ',
                   ' '.join (('feature',
                              'interval_left', 'interval_right',
                              'old_dist', 'new_dist')),
                   '\n')


  def _measure_progress_towards_interval (self,
                                          feature: int,
                                          interval: Interval,
                                          old: Input):
    def aux (new : Input) -> bool:
      acts = self.analyzer.eval_batch (np.array([old, new]),
                                       allow_input_layer = False)
      dimreds = self.dimred_activations (acts)
      old_v = dimreds[0][..., feature : feature + 1].flatten ()[0]
      new_v = dimreds[1][..., feature : feature + 1].flatten ()[0]
      old_dist = interval_dist (interval, old_v)
      new_dist = interval_dist (interval, new_v)
      append_in_file (self.progress_file,
                      ' '.join (str (i) for i in (feature,
                                                  interval[0], interval[1],
                                                  old_dist, new_dist)),
                      '\n')
      return (old_dist - new_dist, new_dist) if old_dist > 0.0 else \
             (0.0, new_dist)   # return 0 if old_v already in interval
    return aux


  # ---


  def _probas (self, p):
    return p.parameters[0] if not isinstance (p.parameters[0], dict) else \
           [ p.parameters[0][i] for i in p.parameters[0] ]


  def _all_marginals (self) -> range:
    return (self._probas (p) for p in self.N.marginal ())



  def _all_cpts (self):
    return (self._probas (j.distribution)
            for j in self.N.states
            if isinstance (j.distribution, ConditionalProbabilityTable))



  def _all_cpts_n_marginals (self) -> range:
    return ((self._probas (j.distribution), self._probas (m))
            for j, m in zip (self.N.states, self.N.marginal ())
            if isinstance (j.distribution, ConditionalProbabilityTable))





  def bfc_coverage (self) -> Coverage:
    """
    Computes the BFCov metric as per the underlying Bayesian Network
    abstraction.
    """
    assert (self.num_test_cases > 0)
    props = sum (np.count_nonzero (np.array(p) >= self.epsilon) / len (p)
                 for p in self._all_marginals ())
    return Coverage (covered = props, total = self.N.node_count ())


  def bfdc_coverage (self) -> Coverage:
    """
    Computes the BFdCov metric as per the underlying Bayesian Network
    abstraction.
    """
    assert (self.num_test_cases > 0)
    # Count 0s (or < epsilons) in all prob. mass functions in the BN
    # abstraction, subject to associated marginal probabilities being
    # > epsilon as well:
    def count_nonepsilons (acc, x):
      (noneps_props, num_cpts), (cpt, marginal) = acc, x
      # p's last column (-1) holds conditional probabilities, whereas
      # the last but one (-2) holds the feature interval index.
      noneps_props += \
        sum (p[-1] >= self.epsilon if marginal[p[-2]] >= self.epsilon else True \
             for p in cpt) \
        / len (cpt)
      return (noneps_props, num_cpts + 1)
    props, num_cpts = reduce (count_nonepsilons, self._all_cpts_n_marginals (),
                              (0, 0))
    return Coverage (covered = props, total = num_cpts) if num_cpts > 0 else \
           Coverage (covered = 1)


  # ---


  def stat_based_train_cv_initializers (self):
    """
    Initializes the criterion based on traininig data.

    Directly uses argument ``bn_abstr_train_size`` and
    ``bn_abstr_test_size`` arguments given to the constructor, and
    optionally computes some scores (based on flags given to the
    constructor as well).
    """
    bn_abstr = ({ 'test': self._score }
                if (self.score_layer_likelihoods or
                    self.report_on_feature_extractions is not None or
                    self.assess_discretized_feature_probas) else {})
    return [{
      **self.bn_abstr_params,
      'name': 'Bayesian Network abstraction',
      'layer_indexes': set ([fl.layer_index for fl in self.flayers]),
      'train': self._discretize_features_and_create_bn_structure,
      **bn_abstr,
      # 'accum_test': self._accum_fit_bn,
      # 'final_test': self._bn_score,
    }]


  def _discretize_features_and_create_bn_structure (self, acts, **kwds):
    """
    Called through :meth:`stat_based_train_cv_initializers` above.
    """

    ts0 = len(acts[self.flayers[0].layer_index])
    cp1 ('| Given training data of size {}'.format (ts0))
    fts = None if self.feat_extr_train_size == 1 \
          else (min(ts0, int (self.feat_extr_train_size))
                if self.feat_extr_train_size > 1
                else int (ts0 * self.feat_extr_train_size))
    if fts is not None:
      p1 ('| Using training data of size {} for feature extraction'.format (fts))

    # First, fit feature extraction and discretizer parameters:
    for fl in self.flayers:
      p1 ('| Extracting and discretizing features for layer {}... '.format (fl))
      x = np.stack([a.flatten () for a in acts[fl.layer_index]], axis = 0)
      tp1 ('Extracting features...')
      if fts is None:
        y = fl.transform.fit_transform (x)
      else:
        # Copying the inputs here as we pass `copy = False` when
        # constructing the pipeline.
        fl.transform.fit (copy.copy (x[:fts]))
        y = fl.transform.transform (x)
      p1 ('| Extracted {} feature{}'.format (y.shape[1], 's' if y.shape[1] > 1 else ''))
      tp1 ('Discretizing features...')
      fl.discr.fit_wrt (x, y, fl.transform, layer = fl, **kwds,
                        outdir = self.outdir)
      del x, y

    self.explained_variance_ratios_ = \
      { str(fl): fl.transform[-1].explained_variance_ratio_.tolist ()
        for fl in self.flayers
        if hasattr (fl.transform[-1], 'explained_variance_ratio_') }

    # Report on explained variance
    for fl in self.explained_variance_ratios_:
      p1 ('| Captured variance ratio for layer {} is {:6.2%}'
          .format (fl, sum (self.explained_variance_ratios_[fl])))

    # Second, fit some distributions with input layer values (NB: well, actually...)
    # Third, contruct the Bayesian Network
    self.N = self._create_bayesian_network ()

    self.fidx2fli = {}
    feature = 0
    for fl in self.flayers:
      for i in range (fl.num_features):
        self.fidx2fli[feature + i] = (fl, i)
      feature += fl.num_features

    # Last, fit the Bayesian Network with given training activations
    # for now, for the purpose of preliminary assessments; the BN will
    # be re-initialized upon the first call to `add_new_test_cases`:
    if self.score_layer_likelihoods or self.assess_discretized_feature_probas:
      self.fit_activations (acts)


  def get_params (self, deep = True):
    p = dict (node_count = self.N.node_count (),
              edge_count = self.N.edge_count (),
              explained_variance_ratios = self.explained_variance_ratios_)
    if deep:
      p['layers'] = [ fl.get_params (deep) for fl in self.flayers ]
    return p


  def _create_bayesian_network (self):
    """
    Actual BN instantiation.
    """

    import gc
    nc = sum (f.num_features for f in self.flayers)
    max_ec = sum (f.num_features * g.num_features
                  for f, g in zip (self.flayers[:-1], self.flayers[1:]))

    tp1 ('| Creating Bayesian Network of {} nodes and a maximum of {} edges...'
         .format (nc, max_ec))
    N = BayesianNetwork (name = 'BN Abstraction')

    fl0 = self.flayers[0]
    nodes = [ DiscretizedInputFeatureNode (fl0, feature)
              for feature in range (fl0.num_features) ]
    N.add_nodes (*(n for n in nodes))

    gc.collect ()
    prev_nodes = nodes
    for fl in self.flayers[1:]:
      nodes = [ DiscretizedHiddenFeatureNode (fl, feature, prev_nodes)
                for feature in range (fl.num_features) ]
      N.add_nodes (*(n for n in nodes))

      for n in nodes:
        for pn in n.prev_nodes_:
          N.add_edge (pn, n)
      tp1 ('| Creating Bayesian Network: {}/{} nodes, {}/{} edges done...'
           .format (N.node_count (), nc, N.edge_count (), max_ec))

      del prev_nodes
      gc.collect ()
      prev_nodes = nodes

    del prev_nodes
    gc.collect ()
    ec = N.edge_count ()
    tp1 ('| Creating Bayesian Network of {} nodes and {} edges: baking...'
         .format (nc, ec))
    N.bake ()
    p1 ('| Created Bayesian Network of {} nodes and {} edges.'
        .format (nc, ec))
    return N


  # ---


  def _score (self, acts, true_labels = None, **kwds):
    """
    Basic scores for manual investigations.
    """

    p1 ('| Given test sample of size {}'
         .format(len(acts[self.flayers[0].layer_index])))

    if (self.score_layer_likelihoods or
        self.report_on_feature_extractions is not None):
      self._score_feature_extractions (acts, true_labels)

    if self.assess_discretized_feature_probas:
      truth = self.dimred_n_discretize_activations (acts)
      self._score_discretized_feature_probas (truth)
      del truth


  def _score_feature_extractions (self, acts, true_labels = None):
    racc = None
    idx = 1
    self.average_log_likelihoods_ = []
    for fl in self.flayers:
      flatacts = fl.flatten_map (acts)

      if self.score_layer_likelihoods:
        tp1 ('| Computing average log-likelihood of test sample for layer {}...'
             .format (fl))
        self.average_log_likelihoods_.append (fl.transform.score (flatacts))
        p1 ('| Average log-likelihood of test sample for layer {} is {}'
            .format (fl, self.average_log_likelihood[-1]))

      if self.report_on_feature_extractions is not None:
        fdimred = self.dimred_activations (acts, (fl,))
        racc = self.report_on_feature_extractions (fl, flatacts, fdimred,
                                                   true_labels, racc)
        del fdimred

      idx += 1
      del flatacts

    if self.close_reports_on_feature_extractions is not None:
      self.close_reports_on_feature_extractions (racc)


  def _score_discretized_feature_probas (self, truth):
    """
    Further scoring the predictive abilites of the BN.
    """

    if self.N.edge_count () == 0:
      p1 ('Warning: BN abstraction has no edge: skipping prediction assessments.')
      return

    features_probas = self._setup_estimate_feature_probas (truth)

    all_floss = []
    self.log_losses = []
    self.classification_reports = []
    first_feature_idx = 0
    for fl in self.flayers:
      floss = []
      for feature in range (fl.num_features):
        flabels = list (range (fl.discr.feature_parts (feature)))
        feature_idx = first_feature_idx + feature
        ftruth = truth[..., feature_idx : feature_idx + 1].flatten ()

        tp1 ('| Computing predictions for feature {} of {}...'.format (feature, fl))
        fprobas = features_probas (feature_idx, 1).flatten ()

        tp1 ('| Computing log loss for feature {} of {}...'.format (feature, fl))
        fpredict_probs = self._all_prediction_probas (fprobas)
        loss = log_loss (ftruth, fpredict_probs, labels = flabels)
        floss.append (loss)
        p1 ('| Log loss for feature {} of {} is {}'.format (feature, fl, loss))

        if self.print_classification_reports:
          p1 ('| Classification report for feature {} of {}:'.format (feature, fl))
          fpreds = [ np.argmax (p) for p in fpredict_probs ]
          self.classification_reports.append(
            classification_report (ftruth, fpreds, labels = flabels))
          print (self.classification_reports[-1])
          del fpreds

        del ftruth, fprobas, fpredict_probs, flabels

      self.log_losses.append((np.min (floss), np.mean (floss),
                              np.std (floss), np.max (floss)))
      all_floss.extend(floss)
      first_feature_idx += fl.num_features

    self.all_log_losses = (np.min (all_floss), np.mean (all_floss),
                           np.std (all_floss), np.max (all_floss))
    del features_probas


  def _setup_estimate_feature_probas (self, truth):
    ytest = truth if truth.dtype == float else np.array (truth, dtype = float)
    ftest = ytest.copy ()
    def estimate_feature_probas (feature, nbfeats):
      ftest[..., feature : feature + nbfeats] = np.nan
      probas = np.array (self.N.predict_proba (ftest, n_jobs = self.bn_abstr_n_jobs))
      ftest[..., feature : feature + nbfeats] = truth[..., feature : feature + nbfeats]
      lprobas = probas[..., feature : feature + nbfeats]
      del probas
      return lprobas
    return (lambda feature, nbfeats: estimate_feature_probas (feature, nbfeats))


  def _prediction_probas (self, p):
    return [ p.parameters[0][i] for i in range (len (p.parameters[0])) ]


  def _all_prediction_probas (self, fprobas):
    return [ self._prediction_probas (p) for p in fprobas ]


  # ----


# ---

class BNcTarget (TestTarget):

  def measure_progress(self, t: Input) -> float:
    """
    Measures how a new input `t` improves towards fulfilling the
    target.  A negative returned value indicates that no progress is
    being achieved by the given input.
    """
    raise NotImplementedError


# ---

class BFcTarget (NamedTuple, BNcTarget):
  fnode: DiscretizedFeatureNode
  feature_part: int
  progress: Callable[[Input], float]
  root_test_idx: int

  def __repr__(self) -> str:
    interval = self.fnode.flayer.discr.part_edges (self.fnode.feature,
                                                   self.feature_part)
    return ('interval {} of feature {} in layer {} (from root test {})'
            .format(interval_repr (interval),
                    self.fnode.feature, self.fnode.flayer, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {}'
            .format(self.fnode.flayer.layer_index,
                    self.fnode.feature, self.feature_part))


  def cover(self, acts) -> None:
    # Do nothing for now; ideally: update some probabilities
    # somewhere.
    pass



  def measure_progress(self, t: Input) -> float:
    progress, new_dist = self.progress (t)
    p1 ('| Progress towards {}: {}\n'
        '| Distance to target interval: {}'
        .format (self, progress, new_dist))
    return progress


# ---


class BFcAnalyzer (Analyzer4RootedSearch):
  """
  Analyzer dedicated to targets of type :class:`BFcTarget`.
  """

  @abstractmethod
  def search_input_close_to(self, x: Input, target: BFcTarget) -> Optional[Tuple[float, Input]]:
    """
    Method specialized for targets of type :class:`BFcTarget`.
    """
    pass



# ---


class BFcCriterion (_BaseBFcCriterion, Criterion4RootedSearch):
  '''
  Some kind of "uniformization" coverage criterion.
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               analyzer: BFcAnalyzer,
               *args,
               **kwds):
    assert isinstance (analyzer, BFcAnalyzer)
    super().__init__(clayers, analyzer = analyzer, *args, **kwds)
    self.ban = { fl: set () for fl in self.flayers }


  def __repr__(self):
    return "BFC"


  def reset (self):
    super().reset ()
    self.ban = { fl: set () for fl in self.flayers }


  def coverage (self) -> Coverage:
    return self.bfc_coverage ()


  # def _all_normalized_marginals (self):
  #   marginals = self.N.marginal ()
  #   tot = sum (len (p.parameters[0]) for p in marginals)
  #   res = [ [ p.parameters[0][i] * len (p.parameters[0]) / tot
  #             for i in range (len (p.parameters[0])) ]
  #           for p in marginals ]
  #   del marginals
  #   return res


  def find_next_rooted_test_target (self) -> Tuple[Input, BFcTarget]:

    # Gather non-epsilon marginal probabilities:
    epsilon_entries = ((fli, ints)
                       for fli, prob in enumerate (self._all_marginals ())
                       for ints in np.where (np.asarray(prob) < self.epsilon))

    res, best_dist = None, np.inf
    for feature, epsilon_intervals in epsilon_entries:
      feature_node = self.N.states[feature]
      fl, flfeature = feature_node.flayer, feature_node.feature
      for feature_interval in epsilon_intervals:

        # Search closest test -that does not fall within target interval (?)-:
        # within_target = self._check_within (feature, feature_interval, verbose = False)
        for ti, v in self.all_tests_close_to (feature, feature_interval):
          if ((feature, feature_interval, ti) in self.ban[fl]#  or
              # within_target (flfeature, self.test_cases[i])
              ):
            continue
          dist = interval_dist (feature_node.interval (feature_interval), v)
          if dist < best_dist:
            best_dist = dist
            res = feature, feature_interval, ti, v

    if res is None:
      raise EarlyTermination ('Unable to find a new candidate input!')

    feature, feature_interval, ti, v = res
    feature_node = self.N.states[feature]
    fl, flfeature = feature_node.flayer, feature_node.feature
    interval = feature_node.interval (feature_interval)
    tp1 ('Selecting root test {} at feature-{}-distance {} from {}, layer {}'
         .format (ti, flfeature, v, interval, fl))
    test = self.test_cases[ti]
    measure_progress = \
        self._measure_progress_towards_interval (feature, interval, test)
    fct = BFcTarget (feature_node, feature_interval,
                     # self._check_within (feature, feature_interval),
                     measure_progress, ti)
    self.ban[fl].add ((feature, feature_interval, ti))

    return test, fct


# ---


class BFDcTarget (NamedTuple, BNcTarget):
  fnode1: DiscretizedHiddenFeatureNode
  feature_part1: int
  flayer0: BFcLayer
  feature_parts0: Sequence[int]
  # sanity_check: Callable[[int, int, Input], bool]
  progress: Callable[[Input], float]
  root_test_idx: int

  def __repr__(self) -> str:
    interval = self.fnode1.flayer.discr.part_edges (self.fnode1.feature,
                                                    self.feature_part1)
    return (('interval {} of feature {} in layer {}, subject to feature'+
             ' intervals {} in layer {} (from root test {})')
            .format(interval_repr (interval),
                    self.fnode1.feature, self.fnode1.flayer,
                    self.feature_parts0, self.flayer0, self.root_test_idx))


  def log_repr(self) -> str:
    return ('#layer: {} #feat: {} #part: {} #conds: {}'
            .format(self.fnode1.flayer.layer_index,
                    self.fnode1.feature, self.feature_part1,
                    self.feature_parts0))


  def cover(self, acts) -> None:
    # Do nothing for now; ideally: update some probabilities
    # somewhere.
    pass


  # def check(self, t: Input) -> bool:
  #   """
  #   Checks whether the target is met.
  #   """
  #   return self.sanity_check (self.fnode1.feature, t)


  def measure_progress(self, t: Input) -> float:
    progress, new_dist = self.progress (t)
    p1 ('| Progress towards {}: {}\n'
        '| Distance to target interval: {}'
        .format (self, progress, new_dist))
    return progress


# ---


class BFDcAnalyzer (Analyzer4RootedSearch):
  """
  Analyzer dedicated to targets of type :class:`BDFcTarget`.
  """

  @abstractmethod
  def search_input_close_to(self, x: Input, target: BFDcTarget) -> Optional[Tuple[float, Input]]:
    """
    Method specialized for targets of type :class:`BFDcTarget`.
    """
    pass



# ---


class BFDcCriterion (_BaseBFcCriterion, Criterion4RootedSearch):
  '''
  Adaptation of MC/DC coverage for partitioned features.
  '''

  def __init__(self,
               clayers: Sequence[CoverableLayer],
               analyzer: BFDcAnalyzer,
               *args,
               **kwds):
    assert isinstance (analyzer, BFDcAnalyzer)
    super().__init__(clayers, analyzer = analyzer, *args, **kwds)
    assert len(self.flayers) >= 2
    self.ban = { fl: set () for fl in self.flayers }


  def __repr__(self):
    return "BFdC"


  def reset (self):
    super().reset ()
    self.ban = { fl: set () for fl in self.flayers }


  def coverage (self) -> Coverage:
    return self.bfdc_coverage ()


  def find_next_rooted_test_target (self) -> Tuple[Input, BFcTarget]:

    # Gather non-epsilon conditional probabilities:
    cpts = [ np.array (cpt) for cpt in self._all_cpts () ]
    epsilon_entries = ((i, fli)
                       for i, cpt in enumerate (cpts)
                       for fli in np.where (cpt[:,-1] < self.epsilon))

    res, best_dist = None, np.inf
    for i, epsilon_intervals in epsilon_entries:
      feature = self.flayers[0].num_features + i
      feature_node = self.N.states[feature]
      fl, flfeature = feature_node.flayer, feature_node.feature

      for fli in epsilon_intervals:
        assert cpts[i][fli, -1] < self.epsilon
        feature_interval = int (cpts[i][fli, -2])

        # Search closest test -that does not fall within target interval (?)-:
        # within_target = self._check_within (feature, feature_interval, verbose = False)
        for ti, v in self.all_tests_close_to (feature, feature_interval):
          if ((feature, feature_interval, ti) in self.ban[fl]#  or
              # within_target (flfeature, self.test_cases[i])
              ):
            continue
          dist = interval_dist (feature_node.interval (feature_interval), v)
          if dist < best_dist:
            best_dist = dist
            res = i, fli, ti, v

    if res is None:
      raise EarlyTermination ('Unable to find a new candidate input!')

    i, fli, ti, v = res
    feature = self.flayers[0].num_features + i
    feature_node = self.N.states[feature]
    feature_interval = int (cpts[i][fli, -2])
    fl, flfeature = feature_node.flayer, feature_node.feature
    fl_prev, _ = self.fidx2fli[feature - flfeature - 1]
    interval = feature_node.interval (feature_interval)
    tp1 ('Selecting root test {} at feature-{}-distance {} from {}, layer {}'
         .format (ti, flfeature, v, interval, fl))
    test = self.test_cases[ti]
    measure_progress = \
        self._measure_progress_towards_interval (feature, interval, test)
    cond_intervals = cpts[i][fli, :-2].astype (int)
    fct = BFDcTarget (feature_node, feature_interval, fl_prev, cond_intervals,
                      # self._check_within (feature, feature_interval),
                      measure_progress, ti)
    self.ban[fl].add ((feature, feature_interval, ti))

    return test, fct



# ---



# def abstract_layerp (li, feats = None, discr = None, layer_indices = []):
#   return (li in discr if discr is not None and isinstance (discr, dict) else
#           li in feats if feats is not None and isinstance (feats, dict) else
#           li in layer_indices)

import builtins

def abstract_layer_features (li, feats = None, discr = None, default = 1):
  if feats is not None:
    if isinstance (feats, (int, float)):
      return feats
    if isinstance (feats, str):
      return builtins.eval(feats, {})(li)
    if isinstance (feats, dict) and li in feats:
      li_feats = feats[li]
      if not isinstance (li_feats, (int, float, str, dict)):
        raise ValueError (
          'feats[{}] should be an int, a string, or a float (got {})'
          .format (li, type (li_feats)))
      return li_feats
    elif isinstance (feats, dict):
      return feats
    raise ValueError (
      'feats should either be a dictonary, an int, a string, or a float (got {})'
      .format (type (feats)))

  # Guess from discr
  if discr is not None:
    if isinstance (discr, dict):
      li_bins = discr[li]
      return (len (li_bins) if isinstance (li_bins, list) else
              li_bins if isinstance (li_bins, int) else
              default)
    elif (isinstance (discr, list) and li < len (discr) and
          isinstance (discr[li], list)):
      return (len (discr[li]))

  return default


def abstract_layer_feature_discretization (l, li, discr = None):
  li_discr = (discr[li] if isinstance (discr, dict) and li in discr else
              discr     if isinstance (discr, dict) else
              discr     if isinstance (discr, int)  else
              discr[li] if isinstance (discr, list) else
              discr(li) if callable (discr) else
              builtins.eval(discr, {})(li) if (isinstance (discr, str) and
                                               discr not in ('binarizer', 'bin')) else
              None)
  if li_discr in (None, 'binarizer', 'bin'):
    p1 ('Using binarizer for layer {.name}'.format (l))
    return FeatureBinarizer ()
  else:
    k = (li_discr if isinstance (li_discr, int) else
         li_discr['n_bins'] if (isinstance (li_discr, dict)
                                and 'n_bins' in li_discr) else
         # TODO: per feature discretization strategy?
         None)
    s = (li_discr['strategy'] if (isinstance (li_discr, dict)
                                  and 'strategy' in li_discr) else
         'quantile')
    extended = (isinstance (li_discr, dict) and 'extended' in li_discr
                and li_discr['extended'])
    extended = extended is not None and extended
    p1 ('Using {}{}discretizer with {} strategy for layer {.name}'
        .format ('extended ' if extended else '',
                 '{}-bin '.format (k) if k is not None else '',
                 s, l))
    cstr = KBinsNOutFeatureDiscretizer if extended else KBinsFeatureDiscretizer
    discr_args = { **(li_discr if isinstance (li_discr, dict) else {}),
      'n_bins': k,
      'encode': 'ordinal',
      'strategy': s,
    }
    if 'extended' in discr_args:
      del discr_args['extended']
    return cstr (**discr_args)


def abstract_layer_setup (l, i, feats = None, discr = None, **kwds):
  options = abstract_layer_features (i, feats, discr)
  if isinstance (options, dict) and 'decomp' in options:
    decomp = options['decomp']
    options = { **options }
    del options['decomp']
  else:
    decomp = 'pca'
  if (decomp == 'pca' and
      (isinstance (options, (int, float)) or options == 'mle')):
    svd_solver = ('arpack' if isinstance (options, int) else
                  'full' if isinstance (options, float) else 'auto')
    options = { 'n_components': options, 'svd_solver': svd_solver }
  # from sklearn.decomposition import IncrementalPCA
  fext = (make_pipeline (StandardScaler (copy = False),
                         PCA (**options, copy = False)) if decomp == 'pca' else
          make_pipeline (FastICA (**options)))
  feature_discretization = abstract_layer_feature_discretization (l, i, discr, **kwds)
  return BFcLayer (layer = l, layer_index = i,
                   transform = fext,
                   discretization = feature_discretization)


# ---


def plot_report_on_feature_extractions (fl, flatacts, fdimred, labels, acc = None):
  if not plt:
    warnings.warn ('Unable to import `matplotlib`: skipping feature extraction plots')
    return

  minlabel, maxlabel = np.min (labels), np.max (labels)
  cmap = plt.get_cmap ('nipy_spectral', maxlabel - minlabel + 1)

  flabel = (lambda feature:
            ('f{} (variance ratio = {:6.2%})'
             .format (feature, fl.transform[-1].explained_variance_ratio_[feature]))
            if hasattr (fl.transform[-1], 'explained_variance_ratio_') else
            ('f'+str(feature)))

  maxfeature = fdimred.shape[1] - 1
  if maxfeature < 1:
    return                              # for now
  feature = 0
  while feature + 1 <= maxfeature:
    fig = plt.figure ()
    if feature + 1 == maxfeature:
      ax = fig.add_subplot (111)
      # plt.subplot (len (self.flayer_transforms), 1, idx)
      ax.scatter(fdimred[:,0], fdimred[:,1], c = labels,
                 s = 2, marker='o', zorder = 10,
                 cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (feature))
      ax.set_ylabel (flabel (feature+1))
      feature_done = 2
      incr = 1
    else:
      ax = fig.add_subplot (111, projection = '3d')
      scat = ax.scatter (fdimred[:, feature], fdimred[:, feature+1],
                         fdimred[:, feature+2], c = labels,
                         s = 2, marker = 'o', zorder = 10,
                         cmap = cmap, vmin = minlabel - .5, vmax = maxlabel + .5)
      ax.set_xlabel (flabel (feature))
      ax.set_ylabel (flabel (feature+1))
      ax.set_zlabel (flabel (feature+2))
      feature_done = 3
      incr = 1 if feature + 1 == maxfeature - 2 else 2
    fig.suptitle ('Features {} of layer {}'
                  .format (tuple (range (feature, feature + feature_done)), fl))
    cb = fig.colorbar (scat, ticks = range (minlabel, maxlabel + 1), label = 'Classes')
    feature += incr
  plt.draw ()


# ---


from engine import setup as engine_setup

def setup (setup_criterion = None,
           test_object = None,
           outdir: OutputDir = None,
           feats = { 'n_components': 2, 'svd_solver': 'randomized' },
           feat_extr_train_size = 1,
           discr = 'bin',
           epsilon = None,
           report_on_feature_extractions = False,
           bn_abstr_train_size = 0.5,
           bn_abstr_test_size = 0.5,
           bn_abstr_n_jobs = None,
           **kwds):

  if setup_criterion is None:
    raise ValueError ('Missing argument `setup_criterion`!')

  setup_layer = (lambda l, i, **kwds: abstract_layer_setup (l, i, feats, discr))
  cover_layers = get_cover_layers (test_object.dnn, setup_layer,
                                   layer_indices = test_object.layer_indices,
                                   exclude_direct_input_succ = False,
                                   exclude_output_layer = False)
  criterion_args \
    = dict (bn_abstr_train_size = bn_abstr_train_size,
            bn_abstr_test_size = bn_abstr_test_size,
            feat_extr_train_size = feat_extr_train_size,
            print_classification_reports = True,
            epsilon = epsilon,
            bn_abstr_n_jobs = bn_abstr_n_jobs,
            outdir = outdir)
  if report_on_feature_extractions:
    criterion_args['report_on_feature_extractions'] = plot_report_on_feature_extractions
    criterion_args['close_reports_on_feature_extractions'] = (lambda _: ploting.show ())

  return engine_setup (test_object = test_object,
                       cover_layers = cover_layers,
                       setup_criterion = setup_criterion,
                       criterion_args = criterion_args,
                       **kwds)

# ---

