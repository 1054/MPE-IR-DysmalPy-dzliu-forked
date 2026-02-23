#!/usr/bin/env python
# coding: utf-8


import os, sys, re, json, copy
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict

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

# get obs and model
gal2 = copy.deepcopy(gal)
obs = gal.observations['OBS']
model = gal.model
if obs.model_data is None:
    gal.create_model_data()
obs_extract, model_out = plotting.extract_1D_2D_obs_from_cube(obs, model, inst_corr=True,
                                           fill_mask=fill_mask)

# Data haven't actually been corrected for instrument LSF yet
# (Note: 1D/2D *models* will be corrected for LSF during plotting,
#        based on the data['inst_corr'] setting)
if obs_extract['extract_1D'].data.data['inst_corr'] and obs_extract['extract_1D'].instrument.lsf is not None:
    inst_corr_sigma = obs_extract['extract_1D'].instrument.lsf.dispersion.to(u.km/u.s).value
    disp_prof_1D = np.sqrt(obs_extract['extract_1D'].data.data['dispersion']**2 - inst_corr_sigma**2 )
    disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
    obs_extract['extract_1D'].data.data['dispersion'] = disp_prof_1D

    if 'filled_mask_data' in obs_extract['extract_1D'].data.__dict__.keys():
        disp_prof_1D = np.sqrt(obs_extract['extract_1D'].data.filled_mask_data.data['dispersion']**2 - inst_corr_sigma**2 )
        disp_prof_1D[~np.isfinite(disp_prof_1D)] = 0.
        obs_extract['extract_1D'].data.filled_mask_data.data['dispersion'] = disp_prof_1D


if obs_extract['extract_2D'].data.data['inst_corr'] and obs_extract['extract_2D'].instrument.lsf is not None:
    inst_corr_sigma = obs_extract['extract_2D'].instrument.lsf.dispersion.to(u.km/u.s).value
    im = obs_extract['extract_2D'].data.data['dispersion'].copy()
    im = np.sqrt(im ** 2 - inst_corr_sigma ** 2)
    im[~np.isfinite(im)] = 0.
    obs_extract['extract_2D'].data.data['dispersion'] = im


# Save 2D as fits
fileout = os.path.join(params['outdir'], 'output_bestfit_data_model_2D.fits')
HDUList = [fits.PrimaryHDU()]
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].data.data['flux'], name='DATA FLUX'))
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].data.data['velocity'], name='DATA VELOCITY'))
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].data.data['dispersion'], name='DATA DISPERSION'))
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].model_data.data['flux'], name='MODEL FLUX'))
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].model_data.data['velocity'], name='MODEL VELOCITY'))
HDUList.append(fits.ImageHDU(data=obs_extract['extract_2D'].model_data.data['dispersion'], name='MODEL DISPERSION'))
if os.path.exists(fileout):
    shutil.move(fileout, fileout+'.backup')
fits.HDUList(HDUList).writeto(fileout)
print('Output to {}'.format(fileout))


# Save 1D as csv
fileout = os.path.join(params['outdir'], 'output_bestfit_data_model_1D.csv')
TableDict = OrderedDict()
TableDict['RADIUS_ARCSEC'] = obs_extract['extract_1D'].data.rarr
TableDict['DATA_FLUX'] = obs_extract['extract_1D'].data.data['flux']
TableDict['DATA_VELOCITY'] = obs_extract['extract_1D'].data.data['velocity']
TableDict['DATA_DISPERSION'] = obs_extract['extract_1D'].data.data['dispersion']
TableDict['MODEL_FLUX'] = obs_extract['extract_1D'].model_data.data['flux']
TableDict['MODEL_VELOCITY'] = obs_extract['extract_1D'].model_data.data['velocity']
TableDict['MODEL_DISPERSION'] = obs_extract['extract_1D'].model_data.data['dispersion']
table = Table(TableDict)
table.meta = {}
table.meta['SLIT_WIDTH'] = obs_extract['extract_1D'].instrument.slit_width
table.meta['SLIT_PA'] = obs_extract['extract_1D'].instrument.slit_pa
if os.path.exists(fileout):
    shutil.move(fileout, fileout+'.backup')
table.write(fileout, overwrite=True)
print('Output to {}'.format(fileout))


