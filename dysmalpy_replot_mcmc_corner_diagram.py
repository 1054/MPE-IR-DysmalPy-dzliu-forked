#!/usr/bin/env python
# coding: utf-8


import os, sys, re, json, copy
import numpy as np

sys.path.insert(1, os.path.abspath(os.path.dirname(__file__)))
import dysmalpy
print('dysmalpy.__path__', dysmalpy.__path__)
print('np.__version__', np.__version__) # need to be < 2.0.0

from dysmalpy.fitting_wrappers import dysmalpy_fit_single
from dysmalpy.fitting_wrappers import utils_io
from dysmalpy import fitting, plotting


if len(sys.argv) <= 1:
    print('Usage: input a *.params file')
    sys.exit()


param_file = sys.argv[1]


# load parameter file
params = utils_io.read_fitting_params(fname=param_file)
params['outdir'] = params['outdir'].rstrip('/')+'/' # ensure_path_trailing_slash

# setup target galaxy
gal, output_options = utils_io.setup_single_galaxy(params=params)

# setup fitter
fitter = utils_io.setup_fitter(params=params)

# set output options
output_options.set_output_options(gal, fitter)
output_options.overwrite = False

# restore a previous fitting
fit_results = fitter.fit(gal, output_options)

# redraw corner plot
param_ranges = {
    'sigma0': [0.5, 35.]
}

fileout = os.path.join(params['outdir'], 'output_mcmc_corner_updated.pdf')

fit_results.plot_corner(gal=gal, param_ranges=param_ranges, fileout=fileout, overwrite=True)

print('Output to {}'.format(fileout))
