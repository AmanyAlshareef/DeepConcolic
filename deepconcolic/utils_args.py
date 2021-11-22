import argparse
import traceback
from .utils_funcs import some, rng_seed
from .utils_io import sys, dir_or_file_in_dir, p1

# ---

def add_named_n_pos_args (parser, posname, flags, required = True, **kwds):
  gp = parser.add_mutually_exclusive_group (required = required)
  gp.add_argument (*flags, dest = posname + '_', **kwds)
  gp.add_argument (posname, nargs = '?', **kwds)

def pp_named_arg (posname):
  def aux (args):
    if hasattr (args, posname + '_'):
      named = getattr (args, posname + '_')
      setattr (args, posname, some (named, getattr (args, posname)))
      delattr (args, posname + '_')
  return aux

def make_select_parser (descr, posname, choices, with_flag = False, **kwds):
  ap = argparse.ArgumentParser (description = descr)
  if with_flag:
    add_named_n_pos_args (ap, posname, (f'--{posname}',),
                          choices = choices.keys (),
                          help = "selected option", **kwds)
  else:
    ap.add_argument (posname, choices = choices.keys (),
                     help = "selected option", **kwds)
  def aux (pp_args = (), **args):
    argsx = vars (ap.parse_args ()) if args is {} or posname not in args else {}
    args = dict (**args, **argsx)
    # pp_args = tuple (pp_named_arg (posname)) + pp_args
    for pp in pp_args: pp (args)
    choices [args[posname]] (**args)
  return ap, aux

def add_verbose_flags (parser, help = 'be more verbose'):
  parser.add_argument ('--verbose', '-v', action = 'store_true', help = help)

# ---

def add_workdir_arg (parser):
  add_named_n_pos_args (parser, 'workdir', ('--workdir', '-d'),
                        type = str, metavar = 'DIR',
                        help = 'work directory')

pp_workdir_arg = pp_named_arg ('workdir')

# ---

def add_abstraction_arg (parser, posname = 'abstraction', short = '-a',
                         help = 'file or directory where the abstraction '
                         '(`abstraction.pkl\' by default) is to be found or '
                         'saved', **kwds):
  add_named_n_pos_args (parser, posname, (f'--{posname}', short),
                        type = str, metavar = 'PKL', help = help, **kwds)

def pp_abstraction_arg (posname = 'abstraction'):
  return pp_named_arg (posname)

abstraction_path = \
  dir_or_file_in_dir ('abstraction.pkl', '.pkl')

# ---

def pp_rng_seed (args):
  """Initialize RNG, with given seed if `args` namespace includes an
  `rng_seed` entry.

  """
  try:
    if 'rng_seed' in args:
      rng_seed (args.rng_seed)
      del args.rng_seed
    else:
      rng_seed (None)
  except ValueError as e:
    sys.exit (f'Invalid argument given for \`--rng-seed\': {e}')

def run_func (func,
              args = None,
              parser = None,
              pp_args = (pp_rng_seed,),
              get_anonargs = ()):
  assert parser is not None
  try:
    args = parser.parse_args () if args is None else args
    for pp in pp_args: pp (args)
    func (*(get (args) for get in get_anonargs), **vars (args))
  except ValueError as e:
    traceback.print_tb (e.__traceback__)
    sys.exit (f'Error: {e}')
  except FileNotFoundError as e:
    sys.exit (f'Error: {e}')
  except KeyboardInterrupt:
    sys.exit ('Interrupted.')

def dispatch_cmd (parser = None, **_):
  def dispatch (*_, cmd = None, **__):
    if cmd is not None:
      assert callable (cmd)
      cmd (*_, **__)
    else:
      parser.print_help ()
      sys.exit (1)
  run_func (dispatch, parser = parser, **_)

# ---
