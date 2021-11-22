from .utils_io import OutputDir, tp1
import os
import tempfile

mpl_backend = os.getenv ('DC_MPL_BACKEND')
mpl_fig_width = float (os.getenv ('DC_MPL_FIG_WIDTH', default = 7.))
mpl_fig_ratio = float (os.getenv ('DC_MPL_FIG_RATIO', default = 1))
mpl_fig_pgf_width = float (os.getenv ('DC_MPL_FIG_PGF_WIDTH', default = 4.7))

default_params = {
  'font.size': 11,
  # 'font.family': 'cmr10',
  'axes.unicode_minus': False,
  'axes.formatter.use_mathtext': True,
  'axes.labelpad': 2,
  'xtick.major.pad': 2.,
  'ytick.major.pad': 2.,
}

png = mpl_backend in ('png', 'PNG')
png_default_figsize = (mpl_fig_width, mpl_fig_width * mpl_fig_ratio)

pgf = mpl_backend in ('pgf', 'PGF')
pgf_output_pgf = True
pgf_output_pdf = True
pgf_default_figsize = (mpl_fig_pgf_width, mpl_fig_pgf_width * mpl_fig_ratio)
pgf_default_params = {
  'font.size': 8,
  # 'font.family': 'cmr10',               # lmodern
  'text.usetex': True,
  'axes.linewidth': .5,
  'axes.unicode_minus': True,           # fix mpl bug in 3.3.0?
  'lines.linewidth': .5,
  'lines.markersize': .2,
  'pgf.texsystem': 'pdflatex',
  'pgf.rcfonts': False,        # don't setup fonts from rc parameters
  # "pgf.preamble": [r"\input{../macro}"]
  'pgf.preamble': "\n".join([
    r"\usepackage[utf8x]{inputenc}",
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amssymb}",
    r"\usepackage{relsize}",
    r"\usepackage{u8chars}",
  ]),
  # 'axes.labelsize': 'small',
  # 'xtick.labelsize': 'x-small',
  # 'ytick.labelsize': 'x-small',
}

# ---

try:
  import matplotlib as mpl
  mpl = mpl
except:
  mpl = None
enabled = mpl is not None

if mpl and pgf:
  mpl.use ('pgf')

try:
  import matplotlib.pyplot as plt
  plt = plt
except:
  plt = None

# ---

def generic_setup (**kwds):
  if plt:
    plt.rcParams.update ({ **default_params, **kwds })

def pgf_setup (**kwds):
  if pgf and plt:
    plt.rcParams.update ({ **pgf_default_params, **kwds })

generic_setup ()
pgf_setup ()

# ---

def _def (f):
  def __aux (*args, figsize = None, figsize_adjust = (1.0, 1.0), **kwds):
    if not plt:
      return None
    figsize = figsize or (pgf_default_figsize if pgf else png_default_figsize)
    figsize = tuple (figsize[i] * figsize_adjust[i] for i in (0, 1))
    return f (*args, figsize = figsize, **kwds)
  return __aux

figure = _def (plt.figure)
subplots = _def (plt.subplots)

# import tikzplotlib

def show (fig = None, outdir = None, basefilename = None,
          subplots_adjust_args = {}, **kwds):
  if plt:
    fig = fig or plt.gcf ()
    if not fig.get_constrained_layout ():
      plt.tight_layout (**{**dict(pad = 0, w_pad = 0.1, h_pad = 0.1),
                           **kwds})
      if fig._suptitle is not None:
        subplots_adjust_args = subplots_adjust_args.copy () or {}
        subplots_adjust_args['top'] = 0.92
      if subplots_adjust_args != {}:
        fig.subplots_adjust (**subplots_adjust_args)
    else:
      fig.set_constrained_layout_pads(**{**dict(w_pad = 0.01, h_pad = 0.01,
                                                hspace=0., wspace=0.),
                                         **kwds})
    if not pgf and not png:
      plt.show ()
    elif basefilename is not None:
      outdir = tempfile.gettempdir () if outdir is None else str (outdir)
      if png:
        f = os.path.join (outdir, basefilename + '.png')
        print ('Outputting {}...'.format (f))
        fig.savefig (f, format='png')
      if pgf and pgf_output_pgf:
        f = os.path.join (outdir, basefilename + '.pgf')
        print ('Outputting {}...'.format (f))
        fig.savefig (f, format='pgf')
      if pgf and pgf_output_pdf:
        f = os.path.join (outdir, basefilename + '.pdf')
        print ('Outputting {}...'.format (f))
        fig.savefig (f, format='pdf')

def _esc (s):
  return s.replace ('_', r'\_').replace ('%', r'\%').replace ('#', r'\#')

def texttt (s):
  return r'\mbox{\smaller\ttfamily ' + _esc (s)  + '}' if pgf else s

def textsc (s):
  return r'\textsc{' + _esc (s) + '}' if pgf else s.upper ()

def text (s):
  if pgf:
    return _esc (s)
  return s

def norm (p):
  import numpy as np
  return r'L_{' + (r'\infty' if p == np.inf else str (int (p))) + r'}'

# # Verbatim copy from: https://jwalton.info/Matplotlib-latex-PGF/
# def set_size(width_pt, fraction=1, subplots=(1, 1)):
#     """Set figure dimensions to sit nicely in our document.

#     Parameters
#     ----------
#     width_pt: float
#             Document width in points
#     fraction: float, optional
#             Fraction of the width which you wish the figure to occupy
#     subplots: array-like, optional
#             The number of rows and columns of subplots.
#     Returns
#     -------
#     fig_dim: tuple
#             Dimensions of figure in inches
#     """
#     # Width of figure (in pts)
#     fig_width_pt = width_pt * fraction
#     # Convert from pt to inches
#     inches_per_pt = 1 / 72.27

#     # Golden ratio to set aesthetic figure height
#     golden_ratio = (5**.5 - 1) / 2

#     # Figure width in inches
#     fig_width_in = fig_width_pt * inches_per_pt
#     # Figure height in inches
#     fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

#     return (fig_width_in, fig_height_in)
