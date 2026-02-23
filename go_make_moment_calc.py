#!/usr/bin/env python
# coding: utf-8

import os, sys, re, json, copy, click, shutil
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
from spectral_cube import SpectralCube


if len(sys.argv) <= 1:
    print('Usage: input a *.fits cube')
    sys.exit()


@click.command()
@click.argument('input_data_cube', type=click.Path(exists=True))
@click.argument('output_name', type=click.Path(exists=False))
def main(input_data_cube, output_name):
    scube = SpectralCube.read(input_data_cube)
    flux = scube.moment0().to(u.Unit(scube.header['BUNIT']) * u.km/u.s).value
    vel = scube.moment1().to(u.km/u.s).value
    disp = scube.linewidth_sigma().to(u.km/u.s).value
    output_dir = os.path.dirname(output_name)
    if output_dir != '' and output_dir != '.' and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if output_name.endswith('.fits'):
        output_name = os.path.splitext(output_name)[0]
    flux_file = output_name+'_flux.fits'
    vel_file = output_name+'_vel.fits'
    disp_file = output_name+'_disp.fits'
    if os.path.exists(flux_file):
        shutil.move(flux_file, flux_file+'.backup')
    fits.PrimaryHDU(flux).writeto(flux_file)
    print('Output to {!r}'.format(flux_file))

    if os.path.exists(vel_file):
        shutil.move(vel_file, vel_file+'.backup')
    fits.PrimaryHDU(vel).writeto(vel_file)
    print('Output to {!r}'.format(vel_file))

    if os.path.exists(disp_file):
        shutil.move(disp_file, disp_file+'.backup')
    fits.PrimaryHDU(disp).writeto(disp_file)
    print('Output to {!r}'.format(disp_file))




if __name__ == '__main__':
    main()

