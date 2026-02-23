#!/usr/bin/env python
# coding: utf-8

import os, sys, re, json, copy, click, shutil, time, datetime
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area, proj_plane_pixel_scales
from reproject import reproject_interp
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling import models as apy_models
from astropy.modeling import fitting as apy_fitting
from regions import Regions # regions >= 0.7 ?

sys.path.insert(1, os.path.abspath(os.path.dirname(__file__)))
import dysmalpy
print('dysmalpy.__path__', dysmalpy.__path__)
print('np.__version__', np.__version__) # need to be < 2.0.0

from dysmalpy.fitting_wrappers import dysmalpy_fit_single
from dysmalpy.fitting_wrappers import utils_io
from dysmalpy import fitting, plotting


if len(sys.argv) <= 1:
    print('Usage: input a *.fits cube')
    sys.exit()



@click.command()
@click.argument('input_data_cube', type=click.Path(exists=True))
@click.option('--name', type=str, default='MyGalaxy', help='galaxy ID/name')
@click.option('--redshift', type=float, default=0.0, help='galaxy ID/name')
@click.option('--fov', type=float, default=None, help='cutout field of view in arcsec, default is no cutout.')
@click.option('--vlsrk', type=float, default=None, help='system center velocity in LSRK frame in km/s, default is no cutout.')
@click.option('--inc', type=float, default=45., help='galaxy initial inclination')
@click.option('--pa', type=float, default=45., help='galaxy initial PA') # the red (receding) side major axis position angle, north is 0
@click.option('--beam-fwhm', type=float, default=None, help='beam (point spread function) FWHM in arcsec, if no beam is in the FITS header')
@click.option('--lsf-sigma', type=float, default=5.0, help='line spread function sigma in km/s')
@click.option('--x-shift', type=float, default=0.0, help='source position x shift in pixels, +x to right')
@click.option('--y-shift', type=float, default=0.0, help='source position y shift in pixels, +y to up')
@click.option('--vel-shift', type=float, default=0.0, help='line center velocity shift in km/s')
@click.option('--do-contsub/--no-contsub', type=bool, is_flag=True, default=True, help='do continuum subtraction for the input cube')
@click.option('--line-mask', type=float, nargs=2, default=None, help='masking line channels for continuum subtraction')
@click.option('--do-params/--no-params', type=bool, is_flag=True, default=True, help='make params file?')
@click.option('--region-mask', type=click.Path(exists=True), default=None, help='A DS9-format region mask for valid area. If a region mask file is set, then --no-auto-mask is set.')
@click.option('--do-auto-mask/--no-auto-mask', type=bool, is_flag=True, default=True, help='make auto mask?')
def main(input_data_cube, name, redshift, fov, vlsrk, inc, pa, beam_fwhm, lsf_sigma, x_shift, y_shift, vel_shift, 
         do_contsub, line_mask, do_params, region_mask, do_auto_mask):

    ## Prepare data, data error and mask cubes

    # We need to make a square data cube, and derive an error cube 
    # if user has not provided one, as well as a mask cube.

    # read input data
    with fits.open(input_data_cube) as hdul:
        beam = None
        if 'BEAMS' in hdul:
            beams = hdul['BEAMS'].data
            beam = [float(np.nanmean(beams['BMAJ'])/3600.), # arcsec->deg
                    float(np.nanmean(beams['BMIN'])/3600.), # arcsec->deg
                    float(np.nanmean(beams['BPA']))]
        i = 0
        iext = -1
        while i < len(hdul):
            if len(hdul[i].data.shape) >= 3:
                data = hdul[i].data
                header = copy.copy(hdul[0].header)
                if i > 0:
                    for key in hdul[i].header:
                        header[key] = hdul[i].header[key]
                iext = i
            i += 1
        if iext == -1:
            raise Exception('Error! No 3D data in the input fits file: {!r}'.format(input_data_cube))
        if beam is None: 
            if ('BMAJ' not in header or 'BMIN' not in header or 'BPA' not in header) and beam_fwhm is None:
                raise Exception('Error! No BMAJ BMIN BPA in the input fits file: {!r}'.format(input_data_cube))
            else:
                beam = [float(header['BMAJ']), 
                        float(header['BMIN']), 
                        float(header['BPA'])]

    if beam_fwhm is not None and beam is not None:
        if np.abs((beam_fwhm/3600.0 - beam[0])/((beam_fwhm/3600.0 + beam[0])/2)) > 0.2:
            raise Exception('The user input --beam-fwhm conflicts with the beam in the FITS header!')
    if beam_fwhm is not None and beam is None:
        beam = [float(beam_fwhm/3600.0), float(beam_fwhm/3600.0), 0.0] # arcsec->deg
    
    print('beam {:.5g} arcsec x {:.5g} arcsec, PA {:.5g} deg'.format(beam[0]*3600, beam[1]*3600, beam[2]))
    header['BMAJ'] = beam[0]
    header['BMIN'] = beam[1]
    header['BPA'] = beam[2]

    # make sure the data is 3D
    i = 1
    ispec = -1
    while i <= header['NAXIS']:
        if header[f'CTYPE{i}'] in ['FREQ', 'VELO', 'VOPT', 'VRAD', 'WAVE']:
            ispec = i
            break
        i += 1
    if ispec == -1:
        raise Exception('Error! The input fits file does not contain spectral axis? CTYPE should be {}'.format(['FREQ', 'VELO', 'VOPT', 'VRAD', 'WAVE']))
    print(f'spectral axis: {ispec}')

    if ispec != 3:
        print(f'data.shape: {data.shape}')
        print(f'Swapping axes: {len(data.shape)-ispec} -> {len(data.shape)-3}')
        data = np.swapaxes(data, len(data.shape)-ispec, len(data.shape)-3)
        print(f'data.shape: {data.shape}')
        for key in ['NAXIS3', 'CTYPE3', 'CUNIT3', 'CRPIX3', 'CRVAL3', 'CDELT3', 
                    'CD1_3', 'CD2_3', 'CD3_3', 'CD3_2', 'CD3_1', 
                    'PC1_3', 'PC2_3', 'PC3_3', 'PC3_2', 'PC3_1']:
            oldkey = key.replace('3', str(ispec))
            if oldkey in header:
                header[key] = header[oldkey]
    
    while header['NAXIS'] > 3:
        data = np.nanmean(data, axis=0)
        i = header['NAXIS']
        for key in [f'NAXIS{i}', f'CTYPE{i}', f'CUNIT{i}', f'CRPIX{i}', f'CRVAL{i}', f'CDELT{i}']:
            if key in header:
                del header[key]
        for j in range(1, i):
            for key in [f'CD{i}_{j}', f'CD{j}_{i}', f'PC{i}_{j}', f'PC{j}_{i}']:
                if key in header:
                    del header[key]
        for key in [f'CD{i}_{i}', f'PC{i}_{i}']:
            if key in header:
                del header[key]
        header['NAXIS'] -= 1

    if header['NAXIS'] != 3 or header['NAXIS3'] <= 1:
        raise Exception('data is not 3D!')
    
    if header['NAXIS3'] <= 1:
        raise Exception('data NAXIS3 is not greater than 1!')

    # convert spectral axis to velocity
    if header['CTYPE3'] == 'FREQ':
        freq_axis = (np.arange(header['NAXIS3'])+1 - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']
        ref_freq = freq_axis[int(len(freq_axis)*0.5)]
        ref_chan = (ref_freq - header['CRVAL3']) / header['CDELT3'] + header['CRPIX3']
        ref_chan = np.round(ref_chan, 4)
        vel_axis = (ref_freq - freq_axis)/ref_freq * 2.99792458e5
        header['CTYPE3'] = 'VELO'
        header['CUNIT3'] = 'km/s'
        header['CRPIX3'] = ref_chan
        header['CRVAL3'] = 0.0
        header['CDELT3'] = float(np.mean(np.diff(vel_axis)))
        for key in ['CD3_3', 'PC3_3']:
            if key in header:
                del header[key]
    elif header['CTYPE3'] in ['VELO', 'VOPT', 'VRAD']:
        vel_axis = (np.arange(header['NAXIS3'])+1 - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']
        if u.Unit(header['CUNIT3']) != u.Unit('km/s'):
            vel_axis = (vel_axis * u.Unit(header['CUNIT3'])).to(u.Unit('km/s')).value
            header['CDELT3'] = (header['CDELT3'] * u.Unit(header['CUNIT3'])).to(u.Unit('km/s')).value
            header['CRVAL3'] = (header['CRVAL3'] * u.Unit(header['CUNIT3'])).to(u.Unit('km/s')).value
            header['CUNIT3'] = 'km/s'
        header['CTYPE3'] = 'VELO'
    else:
        raise NotImplementedError('Converting CTYPE3 {} to velocity is not implemented!'.format(header['CTYPE3']))

    if 'PC1_1' in header and header['PC1_1'] == 1.0:
        for key in ['PC1_1', 'PC1_2', 'PC1_3', 'PC2_1', 'PC2_2', 'PC2_3', 'PC3_1', 'PC3_2', 'PC3_3']:
            if key in header:
                del header[key]
    
    # cut to fov
    nchan, ny, nx = data.shape
    if fov is not None:
        original_data = data.copy()
        original_header = copy.deepcopy(header)
        wcs = WCS(header, naxis=2)
        pixscale = np.sqrt(proj_plane_pixel_area(wcs))*3600.
        x_size = int(np.ceil(fov / pixscale))
        y_size = int(np.ceil(fov / pixscale))
        new_data = np.zeros([nchan, y_size, x_size])
        x_pixsc = pixscale
        y_pixsc = pixscale
        center_RA, center_Dec = wcs.wcs_pix2world([(nx-1)/2.0], [(ny-1)/2.0], 0)
        center_RA, center_Dec = center_RA[0], center_Dec[0]
        cutout_header = fits.Header()
        cutout_header['BITPIX'] = -32
        cutout_header['NAXIS'] = 2
        cutout_header['NAXIS1'] = x_size
        cutout_header['NAXIS2'] = y_size
        cutout_header['CTYPE1'] = 'RA---TAN'
        cutout_header['CTYPE2'] = 'DEC--TAN'
        cutout_header['CUNIT1'] = 'deg'
        cutout_header['CUNIT2'] = 'deg'
        cutout_header['CDELT1'] = -x_pixsc / 3600.0
        cutout_header['CDELT2'] = y_pixsc / 3600.0
        cutout_header['CRPIX1'] = (x_size+1)/2. # 1-based number
        cutout_header['CRPIX2'] = (y_size+1)/2. # 1-based number
        cutout_header['CRVAL1'] = center_RA
        cutout_header['CRVAL2'] = center_Dec
        cutout_header['RADESYS'] = 'ICRS'
        cutout_header['EQUINOX'] = 2000
        #'NAXIS','NAXIS1','NAXIS2','CDELT1','CDELT2','CRPIX1','CRPIX2','CRVAL1','CRVAL2',
        for key in ['BUNIT','BMAJ','BMIN','BPA','TELESCOP','INSTRUME','FILTER','EXPTIME','PA_V3',
                    'DATE-OBS','TIME-OBS','PHOTMODE','PHOTFLAM','PHTFLAM1','PHTFLAM2','PHTRATIO','PHOTFNU','PHOTZPT','PHOTPLAM','PHOTBW',
                    'S_REGION',]:
            if key in header:
                cutout_header[key] = header[key]

        for ichan in range(nchan):
            image = original_data[ichan, :, :]
            cutout_image, cutout_footprint = reproject_interp((image, wcs), cutout_header)
            new_data[ichan, :, :] = cutout_image

        data = new_data

        for key in cutout_header:
            if key in ['BITPIX', 'NAXIS']:
                continue
            header[key] = cutout_header[key]

    nchan, ny, nx = data.shape

    # pad to square
    if nx < ny:
        pad1 = int((ny-nx)/2)
        pad2 = (ny-nx)-pad1
        data_pad = np.full([nchan, ny, ny], fill_value=np.nan)
        data_pad[:, :, pad1:-pad2] = data[:, :, :]
        header['CRPIX1'] = header['CRPIX1'] + pad1
        header['NAXIS1'] = header['NAXIS1'] + pad1 + pad2
    elif nx > ny:
        pad1 = int((nx-ny)/2)
        pad2 = (nx-ny)-pad1
        data_pad = np.full([nchan, nx, nx], fill_value=np.nan)
        data_pad[:, pad1:-pad2, :] = data[:, :, :]
        header['CRPIX2'] = header['CRPIX2'] + pad1
        header['NAXIS2'] = header['NAXIS2'] + pad1 + pad2
    else:
        data_pad = data

    # shift to LSRK
    if vlsrk is not None:
        header['CRVAL3'] -= vlsrk
        vel_axis -= vlsrk

    # check overall spectrum
    # spec_data = np.nanmean(data_pad, axis=(1,2))
    # gauss_model = apy_models.Gaussian1D(amplitude=np.nanmax(spec_data), mean=0.0, stddev=50.0)
    # gauss_fitter = apy_fitting.LevMarLSQFitter()
    # gauss_bestfit = gauss_fitter(gauss_model, vel_axis, spec_data)
    # gauss_fwhz = gauss_bestfit.stddev * 2.35482
    # gauss_center = gauss_bestfit.mean
    # cont_mask = (vel_axis < gauss_center - gauss_fwhz)
    if do_contsub:
        if line_mask is None:
            line_mask = [-120, 120] # km/s
        cont_mask = np.logical_or(vel_axis < line_mask[0], vel_axis > line_mask[1])
        if np.count_nonzero(cont_mask) == 0:
            raise Exception('Error! There is no continuum channel after line masking!')
        cont_mask2D = np.repeat(cont_mask[:, np.newaxis], data_pad.shape[1], axis=1)
        cont_mask3D = np.repeat(cont_mask2D[:, :, np.newaxis], data_pad.shape[2], axis=2)
        cont_image = np.nanmean(data_pad, where=cont_mask3D, axis=0)
        cont_image3D = np.repeat(cont_image[np.newaxis, :, :], data_pad.shape[0], axis=0)
        data_pad = data_pad - cont_image3D
        print('Done continuum subtraction with line mask {} to {} km/s'.format(line_mask[0], line_mask[1]))
        header['CONTSUB'] = True
        if os.path.exists('fdata_cont.fits'):
            shutil.move('fdata_cont.fits', 'fdata_cont.fits.backup')
        fits.PrimaryHDU(cont_image).writeto('fdata_cont.fits')
        print('Output to {!r}'.format('fdata_cont.fits'))

    # iterate over channel images and estiamte rms and mask
    data_err = np.zeros(data_pad.shape) + np.inf
    data_mask = np.isfinite(data_pad)
    if region_mask is not None:
        print('Using user-input region mask: {!r}'.format(region_mask))
        region_list = Regions.read(region_mask, format='ds9')
        region_mask_array = np.full(data_pad.shape[1:], dtype=bool, fill_value=False)
        wcs = WCS(header, naxis=2)
        for region_obj in region_list:
            region_mask_image = region_obj.to_pixel(wcs).to_mask().to_image(region_mask_array.shape)
            if region_mask_image is not None:
                region_mask_image = region_mask_image.astype(bool)
                region_mask_array = np.logical_or(region_mask_array, region_mask_image)
    else:
        region_list = None
        if do_auto_mask:
            print('Making auto mask')
        else:
            print('No auto mask, only NaN mask')
    for i in range(nchan):
        channel_image = data_pad[i, :, :]
        # estimate error rms
        mean, med, sig = sigma_clipped_stats(channel_image)
        channel_mask = (channel_image-med) > 3.0*sig
        channel_mask = convolve(channel_mask.astype(int).astype(float), Gaussian2DKernel(1.0)) > 0.25 # broaden the mask
        mean, med, sig = sigma_clipped_stats(channel_image, mask = channel_mask)
        data_err[i, :, :] = sig # if you have pbcorr, correct for it here!
        if region_mask is not None:
            data_mask[i, :, :] = np.logical_and(data_mask[i, :, :], region_mask_array)
        elif do_auto_mask:
            # build a broad mask for emission area
            channel_mask = (channel_image-med) > 2.0*sig
            #channel_mask = (channel_image-med) > 1.0*sig # 20250928
            channel_mask = convolve(channel_mask.astype(int).astype(float), Gaussian2DKernel(1.5)) > 0.25 # broaden the mask
            data_mask[i, :, :] = np.logical_and(data_mask[i, :, :], channel_mask)
    data_mask = data_mask.astype(int)

    # output files
    if os.path.exists('fdata.fits'):
        print('Backing up {!r} as {!r}'.format('fdata.fits', 'fdata.fits.backup'))
        shutil.move('fdata.fits', 'fdata.fits.backup')
    fits.PrimaryHDU(data_pad, header).writeto('fdata.fits', overwrite=True)
    print('Output to {!r}'.format('fdata.fits'))
    if os.path.exists('fdata_err.fits'):
        print('Backing up {!r} as {!r}'.format('fdata_err.fits', 'fdata_err.fits.backup'))
        shutil.move('fdata_err.fits', 'fdata_err.fits.backup')
    fits.PrimaryHDU(data_err, header).writeto('fdata_err.fits', overwrite=True)
    print('Output to {!r}'.format('fdata_err.fits'))
    if os.path.exists('fdata_mask.fits'):
        print('Backing up {!r} as {!r}'.format('fdata_mask.fits', 'fdata_mask.fits.backup'))
        shutil.move('fdata_mask.fits', 'fdata_mask.fits.backup')
    fits.PrimaryHDU(data_mask, header).writeto('fdata_mask.fits', overwrite=True)
    print('Output to {!r}'.format('fdata_mask.fits'))



    ## Update parameter file with cube information

    # get cube information
    pixscale = header['CDELT2']*3600. # or sometimes 'CD2_2'
    fov_npix = header['NAXIS2']
    print('pixscale', pixscale)

    nspec = header['NAXIS3']
    spec_type = 'velocity' # always velocity
    spec_start = (1 - header['CRPIX3']) * header['CDELT3'] + header['CRVAL3']
    spec_step = header['CDELT3']

    #line_sigma = lsf_sigma # line spread function sigma in km/s, for ALMA, it's very small, ~original channel width
    psf_fwhm = header['BMAJ']*3600. # point spread function FWHM in arcsec
    psf_fwhm_major = header['BMAJ']*3600. # point spread function FWHM major axis in arcsec
    psf_fwhm_minor = header['BMIN']*3600. # point spread function FWHM minor axis in arcsec
    psf_PA = header['BPA'] # point spread function PA in arcsec

    #vel_shift = 0.0
    inc_min = np.round(np.rad2deg(np.asin(max(0.0, np.sin(np.deg2rad(inc)) - 0.3))), 0)
    pa_min = pa - 15.
    x_shift_min = x_shift - 5.
    y_shift_min = y_shift - 5.
    vel_shift_min = vel_shift - 10.
    inc_max = np.round(np.rad2deg(np.asin(min(1.0, np.sin(np.deg2rad(inc)) + 0.3))), 0)
    pa_max = pa + 15.
    x_shift_max = x_shift + 5.
    y_shift_max = y_shift + 5.
    vel_shift_max = vel_shift + 10.

    # 
    if not do_params:
        return

    # 
    param_setup_target = f"""
# ------------------
# Target Setup
# ------------------
galID,        {name:<15s}  # Name of your object
z,            {redshift:<15.6g}  # Redshift
data_dir,     .                # Leave it empty if the data files are in the current directory
outdir,       out_dir/         # Output directory
fdata_cube,   fdata.fits       # data cube
fdata_err,    fdata_err.fits   # data error cube
fdata_mask,   fdata_mask.fits  # mask cube, valid pixels have mask > 0
"""

    # 
    param_setup_data = f"""
# ------------------
# Instrument Setup
# ------------------
pixscale,      {pixscale:<15.4f}  # Pixel scale in arcsec/pixel
fov_npix,      {fov_npix:<15d}  # Number of pixels on a side of model cube
spec_type,     {spec_type:<15s}  # DON'T CHANGE!
spec_start,    {spec_start:<15.4f}  # Starting value for spectral axis       // generally don't change
spec_step,     {spec_step:<15.4f}  # Step size for spectral axis in km/s    // generally don't change
nspec,         {nspec:<15d}  # Number of spectral steps               // generally don't change

# ---------
# LSF Setup
# ---------
use_lsf,       True             # True/False if using an LSF
sig_inst_res,  {lsf_sigma:<15.2f}  # Instrumental dispersion in km/s

# ---------
# PSF Setup
# ---------
psf_type,         Gaussian      # Gaussian, Moffat, or DoubleGaussian
psf_fwhm,         {psf_fwhm:<12.2f}  # PSF FWHM in arcsecs
psf_fwhm_major,   {psf_fwhm_major:<12.2f}  # PSF major axis FWHM in arcsecs
psf_fwhm_minor,   {psf_fwhm_minor:<12.2f}  # PSF minor axis FWHM in arcsecs
psf_PA,           {psf_PA:<12.2f}  # PA of PSF major axis, in deg E of N. (0=N, 90=E)
psf_beta,         -99.          # Beta parameter for a Moffat PSF
"""

    # 
    param_setup_models = f"""
# ------------------
# Model Setup
# ------------------

## SETUP COMPONENT LIST
components_list,         disk+bulge  const_disp_prof  geometry  zheight_gaus halo
light_components_list,   disk

## BARYONS
### Parameter values
total_mass,           11.0       # Total mass of disk and bulge log(Msun)
bt,                   0.3        # Bulge-to-Total Ratio
r_eff_disk,           8.0        # Effective radius of disk in kpc
n_disk,               1.0        # Sersic index for disk
invq_disk,            5.0        # disk scale length to zheight ratio for disk
r_eff_bulge,          1.0        # Effective radius of bulge in kpc
n_bulge,              4.0        # Sersic index for bulge
invq_bulge,           1.0        # disk scale length to zheight ratio for bulge

### Parameter fixed?
total_mass_fixed,     False
r_eff_disk_fixed,     False
bt_fixed,             True
n_disk_fixed,         True
r_eff_bulge_fixed,    True
n_bulge_fixed,        True

### Parameter bounds. Lower and upper bounds
total_mass_bounds,    9.5  13.0
bt_bounds,            0.0  0.9
r_eff_disk_bounds,    0.1  30.0
n_disk_bounds,        1.0  4.0
r_eff_bulge_bounds,   1.0  5.0
n_bulge_bounds,       2.0  8.0

## DARK MATTER
### Parameter values
halo_profile_type,    NFW
mvirial,              11.5       # Halo virial mass in log(Msun)
halo_conc,            5.0        # Halo concentration parameter
fdm,                  0.3        # Dark matter fraction at r_eff_disk

### Parameter fixed?
mvirial_fixed,        False
halo_conc_fixed,      True
fdm_fixed,            False

### Parameter bounds
mvirial_bounds,       9.5 14.0
halo_conc_bounds,     2.0 10.0
fdm_bounds,           0.0 1.0

### Parameter ties (linking one parameter to another by some equations)
mvirial_tied,         False     # for NFW, mvirial_tied=True determines Mvirial from fDM (+baryons)
fdm_tied,             True      # for NFW, fdm_tied=True determines fDM from Mvirial (+baryons)

## DISPERSION PROFILE
sigma0,               20.0      # Constant intrinsic dispersion value
sigma0_fixed,         False
sigma0_bounds,        5.0 300.0

## ZHEIGHT PROFILE
sigmaz,               0.5       # Gaussian width of the galaxy in z, in kpc
sigmaz_fixed,         True
sigmaz_bounds,        0.1 2.0

## GEOMETRY
### Parameter values
inc,                  {inc:<8.2f}  # Inclination of the galaxy, 0=face-on, 90=edge-on
pa,                   {pa:<8.2f}  # Position angle of the major axis of the galaxy
xcenter,              None
ycenter,              None
xshift,               {x_shift:<8.2f}  # pixels
yshift,               {y_shift:<8.2f}  # pixels
vel_shift,            {vel_shift:<8.2f}  # km/s

### Parameter fixed?
inc_fixed,            False
pa_fixed,             False
xshift_fixed,         False
yshift_fixed,         False
vel_shift_fixed,      False

### Parameter bounds.
inc_bounds,           {inc_min:.2f} {inc_max:.2f}
pa_bounds,            {pa_min:.2f} {pa_max:.2f}
xshift_bounds,        {x_shift_min:.2f} {x_shift_max:.2f}  # pixels
yshift_bounds,        {y_shift_min:.2f} {y_shift_max:.2f}  # pixels
vel_shift_bounds,     {vel_shift_min:.2f} {vel_shift_max:.2f}  # km/s

### Parameter priors and stddev for MCMC
inc_prior,            sine_gaussian   # flat or sine_gaussian
pa_prior,             gaussian
xshift_prior,         gaussian
yshift_prior,         gaussian
vel_shift_prior,      gaussian
inc_stddev,           0.2
pa_stddev,            5.0
xshift_stddev,        0.5
yshift_stddev,        0.5
vel_shift_stddev,     3.0

## MISC SETTINGS
adiabatic_contract,   False     # Apply adiabatic contraction?
pressure_support,     True      # Apply assymmetric drift correction?
noord_flat,           True      # Apply Noordermeer flattenning?
oversample,           1         # Spatial oversample factor
oversize,             1         # Oversize factor
zcalc_truncate,       True      # Truncate in zgal direction when calculating or not
moment_calc,          False     # If True, use moments, otherwise perform 1d Gaussian fitting
n_wholepix_z_min,     3         # Minimum number of whole pixels in zgal dir, if zcalc_truncate=True
slit_width,           15.0
slit_pa,              {pa:.2f}
profile1d_type,       circ_ap_cube
"""

    # 
    param_setup_fitting = f"""
# ----------------------
# Fitting Method Setup
# ----------------------
fit_method,           mcmc
linked_posteriors,    total_mass mvirial sigma0
nWalkers,             50
nCPUs,                50
nBurn,                50
nSteps,               1000
do_plotting,          True      # Produce all output plots?
maxiter,              30        # Maximum number of iterations before mpfit quits
fitdispersion,        True
overwrite,            True
"""


    # backup 'fit.params'
    if os.path.exists('fit.params'):
        if os.path.exists('fit.params.backup'):
            print('Backing up {!r} as {!r}'.format('fit.params.backup', 'fit.params.backup.backup'))
            shutil.move('fit.params.backup', 'fit.params.backup.backup')
        print('Backing up {!r} as {!r}'.format('fit.params', 'fit.params.backup'))
        shutil.move('fit.params', 'fit.params.backup')

    # append param_file_content_new to file
    with open('fit.params', 'w') as fp:
        fp.write(param_setup_target)
    with open('fit.params', 'a') as fp:
        fp.write(param_setup_data)
    with open('fit.params', 'a') as fp:
        fp.write(param_setup_models)
    with open('fit.params', 'a') as fp:
        fp.write(param_setup_fitting)




if __name__ == '__main__':
    main()

